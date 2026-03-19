-- ============================================================
-- Полная схема БД системы автоматизации обработки звонков
-- Порядок создания с учётом внешних ключей
-- ============================================================

-- ------------------------------------------------------------
-- 1. Пользователи системы (создаётся первой — на неё ссылаются другие)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id          BIGSERIAL    PRIMARY KEY,
    username    VARCHAR(64)  NOT NULL UNIQUE,
    password    VARCHAR(128) NOT NULL,
    role        VARCHAR(16)  NOT NULL DEFAULT 'operator'
                CHECK (role IN ('operator', 'admin')),
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

COMMENT ON TABLE  users          IS 'Учётные записи пользователей системы';
COMMENT ON COLUMN users.role     IS 'operator — обработка звонков; admin — управление моделью и пользователями';
COMMENT ON COLUMN users.password IS 'Хэш пароля (bcrypt)';


-- ------------------------------------------------------------
-- 2. Аудиозаписи звонков
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audio_records (
    id              BIGSERIAL    PRIMARY KEY,
    call_id         VARCHAR(128) NOT NULL UNIQUE,
    filename        VARCHAR(512) NOT NULL,
    file_path       VARCHAR(1024),
    format          VARCHAR(16)  NOT NULL DEFAULT 'wav',
    duration_sec    DOUBLE PRECISION,
    sample_rate     INTEGER      NOT NULL DEFAULT 16000,
    channels        SMALLINT     NOT NULL DEFAULT 1,
    source_system   VARCHAR(64)  NOT NULL DEFAULT 'ip_pbx',
    agent_id        VARCHAR(128),
    recorded_at     TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_call_id    ON audio_records (call_id);
CREATE INDEX IF NOT EXISTS idx_audio_agent_id   ON audio_records (agent_id);
CREATE INDEX IF NOT EXISTS idx_audio_recorded   ON audio_records (recorded_at DESC);

COMMENT ON TABLE  audio_records               IS 'Аудиозаписи телефонных звонков, поступающих из IP-АТС';
COMMENT ON COLUMN audio_records.call_id       IS 'Уникальный идентификатор звонка, присвоенный IP-АТС';
COMMENT ON COLUMN audio_records.duration_sec  IS 'Длительность записи в секундах';
COMMENT ON COLUMN audio_records.sample_rate   IS 'Частота дискретизации (Гц), после нормализации — 16000';
COMMENT ON COLUMN audio_records.source_system IS 'Система-источник: ip_pbx, manual_upload и др.';


-- ------------------------------------------------------------
-- 3. Транскрипты
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transcripts (
    id                  BIGSERIAL    PRIMARY KEY,
    call_id             VARCHAR(128) NOT NULL UNIQUE
                        REFERENCES audio_records (call_id) ON DELETE CASCADE,
    full_text           TEXT         NOT NULL DEFAULT '',
    segments_json       JSONB        NOT NULL DEFAULT '[]',
    role_mapping_json   JSONB        NOT NULL DEFAULT '{}',
    recognition_quality DOUBLE PRECISION,
    diarization_mode    VARCHAR(32)  NOT NULL DEFAULT 'pyannote',
    processing_time_sec DOUBLE PRECISION,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcripts_call_id ON transcripts (call_id);

COMMENT ON TABLE  transcripts                      IS 'Результаты транскрибации и диаризации звонков (WhisperX)';
COMMENT ON COLUMN transcripts.full_text            IS 'Полный текст диалога, объединённый из сегментов';
COMMENT ON COLUMN transcripts.segments_json        IS 'Массив сегментов: [{start, end, speaker, role, text}]';
COMMENT ON COLUMN transcripts.role_mapping_json    IS 'Маппинг speaker_id → роль (агент / звонящий)';
COMMENT ON COLUMN transcripts.recognition_quality  IS 'Интегральная оценка качества распознавания (0–1)';
COMMENT ON COLUMN transcripts.diarization_mode     IS 'Бэкенд диаризации: pyannote или nemo';


-- ------------------------------------------------------------
-- 4. Результаты классификации (NLP / маршрутизация)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS classification_results (
    id                  BIGSERIAL    PRIMARY KEY,
    call_id             VARCHAR(128) NOT NULL UNIQUE
                        REFERENCES audio_records (call_id) ON DELETE CASCADE,
    intent_id           VARCHAR(128) NOT NULL,
    intent_confidence   DOUBLE PRECISION NOT NULL,
    priority            VARCHAR(32)  NOT NULL DEFAULT 'medium'
                        CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    suggested_group     VARCHAR(128) NOT NULL,
    auto_routed         BOOLEAN      NOT NULL DEFAULT FALSE,
    model_version       VARCHAR(64),
    raw_scores_json     JSONB        NOT NULL DEFAULT '{}',
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cls_call_id   ON classification_results (call_id);
CREATE INDEX IF NOT EXISTS idx_cls_intent_id ON classification_results (intent_id);
CREATE INDEX IF NOT EXISTS idx_cls_priority  ON classification_results (priority);
CREATE INDEX IF NOT EXISTS idx_cls_group     ON classification_results (suggested_group);

COMMENT ON TABLE  classification_results                  IS 'Результаты классификации интентов моделью ruBERT';
COMMENT ON COLUMN classification_results.intent_id        IS 'Идентификатор интента (напр. sales.service_purchase)';
COMMENT ON COLUMN classification_results.intent_confidence IS 'Уверенность модели (0–1); порог автоматики — 0.85';
COMMENT ON COLUMN classification_results.auto_routed      IS 'TRUE — тикет создан автоматически, FALSE — передан оператору';
COMMENT ON COLUMN classification_results.raw_scores_json  IS 'Сырые скоры по всем интентам для аудита';


-- ------------------------------------------------------------
-- 5. Тикеты обращений
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tickets (
    id                BIGSERIAL    PRIMARY KEY,
    ticket_id         VARCHAR(64)  NOT NULL UNIQUE,
    external_id       VARCHAR(128) NOT NULL,
    call_id           VARCHAR(128) NOT NULL
                      REFERENCES audio_records (call_id) ON DELETE RESTRICT,
    title             TEXT         NOT NULL,
    description       TEXT         NOT NULL DEFAULT '',
    priority          VARCHAR(32)  NOT NULL DEFAULT 'medium'
                      CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    status            VARCHAR(32)  NOT NULL DEFAULT 'open',
    assignee_type     VARCHAR(32)  NOT NULL DEFAULT 'group',
    assignee_id       VARCHAR(128) NOT NULL DEFAULT 'default_support',
    intent_id         VARCHAR(128),
    intent_confidence DOUBLE PRECISION,
    entities_json     JSONB        NOT NULL DEFAULT '{}',
    url               VARCHAR(512),
    system            VARCHAR(32)  NOT NULL DEFAULT 'mock',
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tickets_call_id    ON tickets (call_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status     ON tickets (status);
CREATE INDEX IF NOT EXISTS idx_tickets_priority   ON tickets (priority);
CREATE INDEX IF NOT EXISTS idx_tickets_assignee   ON tickets (assignee_id);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tickets_intent_id  ON tickets (intent_id);

COMMENT ON TABLE  tickets                  IS 'Тикеты обращений, создаваемые по результатам обработки звонков';
COMMENT ON COLUMN tickets.ticket_id        IS 'Внутренний UUID тикета в данной системе';
COMMENT ON COLUMN tickets.external_id      IS 'Идентификатор тикета во внешней системе (Jira, Redmine и др.)';
COMMENT ON COLUMN tickets.entities_json    IS 'Извлечённые сущности: {persons, phones, emails, dates, ...}';
COMMENT ON COLUMN tickets.system           IS 'Целевая тикет-система: mock, jira, redmine и др.';


-- ------------------------------------------------------------
-- 6. Обратная связь операторов (датасет для дообучения)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_feedback (
    id                  BIGSERIAL    PRIMARY KEY,
    feedback_id         VARCHAR(128) NOT NULL UNIQUE,
    call_id             VARCHAR(128) NOT NULL
                        REFERENCES audio_records (call_id) ON DELETE CASCADE,
    source_filename     VARCHAR(512),
    decision            VARCHAR(16)  NOT NULL
                        CHECK (decision IN ('accepted', 'rejected')),
    error_type          VARCHAR(32)  NOT NULL DEFAULT 'none'
                        CHECK (error_type IN ('none','wrong_intent','wrong_priority','wrong_group','other')),
    transcript_text     TEXT         NOT NULL DEFAULT '',
    ai_intent_id        VARCHAR(128),
    ai_confidence       DOUBLE PRECISION,
    ai_priority         VARCHAR(32),
    ai_group            VARCHAR(128),
    final_intent_id     VARCHAR(128),
    final_priority      VARCHAR(32),
    final_group         VARCHAR(128),
    auto_learn_applied  BOOLEAN      NOT NULL DEFAULT FALSE,
    auto_learn_message  TEXT,
    operator_id         BIGINT       REFERENCES users (id) ON DELETE SET NULL,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_call_id  ON model_feedback (call_id);
CREATE INDEX IF NOT EXISTS idx_feedback_decision ON model_feedback (decision);
CREATE INDEX IF NOT EXISTS idx_feedback_error    ON model_feedback (error_type);
CREATE INDEX IF NOT EXISTS idx_feedback_operator ON model_feedback (operator_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created  ON model_feedback (created_at DESC);

COMMENT ON TABLE  model_feedback                    IS 'Обратная связь операторов — датасет для дообучения ruBERT';
COMMENT ON COLUMN model_feedback.decision           IS 'accepted — оператор согласен, rejected — исправил предсказание';
COMMENT ON COLUMN model_feedback.error_type         IS 'Тип ошибки модели при decision=rejected';
COMMENT ON COLUMN model_feedback.ai_intent_id       IS 'Исходное предсказание модели (интент)';
COMMENT ON COLUMN model_feedback.final_intent_id    IS 'Исправленное оператором значение интента';
COMMENT ON COLUMN model_feedback.auto_learn_applied IS 'Флаг: пример уже добавлен в обучающую выборку';


-- ------------------------------------------------------------
-- 7. Журнал уведомлений
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS notifications (
    id            BIGSERIAL    PRIMARY KEY,
    call_id       VARCHAR(128) NOT NULL
                  REFERENCES audio_records (call_id) ON DELETE CASCADE,
    ticket_id     VARCHAR(64)  REFERENCES tickets (ticket_id) ON DELETE SET NULL,
    channel       VARCHAR(32)  NOT NULL
                  CHECK (channel IN ('email', 'chat', 'log')),
    destination   VARCHAR(256),
    success       BOOLEAN      NOT NULL DEFAULT FALSE,
    error_message TEXT,
    sent_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_notifications_call_id ON notifications (call_id);
CREATE INDEX IF NOT EXISTS idx_notifications_success ON notifications (success);

COMMENT ON TABLE  notifications              IS 'Журнал отправленных уведомлений исполнителям по тикетам';
COMMENT ON COLUMN notifications.channel     IS 'Канал доставки: email, chat (корп. мессенджер) или log';
COMMENT ON COLUMN notifications.destination IS 'Адресат уведомления: email-адрес, ID чата и т.п.';
