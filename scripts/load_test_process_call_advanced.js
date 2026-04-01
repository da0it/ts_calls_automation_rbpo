import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

function env(name, defaultValue = '') {
  const value = (__ENV[name] || '').trim();
  return value || defaultValue;
}

function requireEnv(name) {
  const value = env(name);
  if (!value) {
    throw new Error(`Missing required env: ${name}`);
  }
  return value;
}

function numEnv(name, defaultValue) {
  const raw = env(name, String(defaultValue));
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`Env ${name} must be numeric, got: ${raw}`);
  }
  return value;
}

function buildScenario() {
  const workloadModel = env('WORKLOAD_MODEL', 'closed').toLowerCase();
  if (workloadModel === 'open') {
    return {
      name: 'process_call_open_model',
      scenario: {
        executor: 'ramping-arrival-rate',
        startRate: numEnv('START_RATE', 1),
        timeUnit: env('ARRIVAL_TIME_UNIT', '1m'),
        preAllocatedVUs: numEnv('PRE_ALLOCATED_VUS', 8),
        maxVUs: numEnv('MAX_VUS', 32),
        stages: [
          { duration: env('WARMUP_DURATION', '1m'), target: numEnv('WARMUP_RATE', 1) },
          { duration: env('RAMP_UP_1', '2m'), target: numEnv('TARGET_RATE_1', 2) },
          { duration: env('RAMP_UP_2', '3m'), target: numEnv('TARGET_RATE_2', 4) },
          { duration: env('SOAK', '5m'), target: numEnv('TARGET_RATE_2', 4) },
          { duration: env('RAMP_DOWN', '1m'), target: 0 },
        ],
        gracefulStop: env('GRACEFUL_STOP', '30s'),
      },
    };
  }

  return {
    name: 'process_call_closed_model',
    scenario: {
      executor: 'ramping-vus',
      startVUs: numEnv('START_VUS', 1),
      stages: [
        { duration: env('WARMUP_DURATION', '1m'), target: numEnv('WARMUP_VUS', 1) },
        { duration: env('RAMP_UP_1', '2m'), target: numEnv('TARGET_VUS_1', 2) },
        { duration: env('RAMP_UP_2', '3m'), target: numEnv('TARGET_VUS_2', 4) },
        { duration: env('SOAK', '5m'), target: numEnv('TARGET_VUS_2', 4) },
        { duration: env('RAMP_DOWN', '1m'), target: 0 },
      ],
      gracefulRampDown: env('GRACEFUL_RAMP_DOWN', '30s'),
    },
  };
}

function parseAudioFixtures() {
  const raw = env('AUDIO_FILES');
  if (!raw) {
    throw new Error('Set AUDIO_FILES to one or more comma-separated audio paths.');
  }
  return raw
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .map((path) => ({
      path,
      name: path.split('/').pop() || 'audio.wav',
      bytes: open(path, 'b'),
    }));
}

function safeJson(response) {
  try {
    return response.json();
  } catch (error) {
    return null;
  }
}

function safeNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function metricValue(data, metricName, fieldName) {
  const metric = data.metrics[metricName];
  if (!metric || !metric.values) {
    return null;
  }
  const value = metric.values[fieldName];
  return Number.isFinite(value) ? value : null;
}

function fmt(value, digits = 3, suffix = '') {
  if (!Number.isFinite(value)) {
    return '-';
  }
  return `${value.toFixed(digits)}${suffix}`;
}

function buildTextSummary(data) {
  const lines = [];
  lines.push('Load test summary');
  lines.push('');
  lines.push(`http_req_failed rate: ${fmt(metricValue(data, 'http_req_failed', 'rate'), 4)}`);
  lines.push(`checks rate: ${fmt(metricValue(data, 'checks', 'rate'), 4)}`);
  lines.push(`http_req_duration avg/p95/p99 (ms): ${fmt(metricValue(data, 'http_req_duration', 'avg'), 1)} / ${fmt(metricValue(data, 'http_req_duration', 'p(95)'), 1)} / ${fmt(metricValue(data, 'http_req_duration', 'p(99)'), 1)}`);
  lines.push(`orchestrator_total_sec avg/p95/p99: ${fmt(metricValue(data, 'orchestrator_total_sec', 'avg'), 3)} / ${fmt(metricValue(data, 'orchestrator_total_sec', 'p(95)'), 3)} / ${fmt(metricValue(data, 'orchestrator_total_sec', 'p(99)'), 3)}`);
  lines.push(`transcription_sec p95: ${fmt(metricValue(data, 'stage_transcription_sec', 'p(95)'), 3)}`);
  lines.push(`routing_sec p95: ${fmt(metricValue(data, 'stage_routing_sec', 'p(95)'), 3)}`);
  lines.push(`entity_extraction_sec p95: ${fmt(metricValue(data, 'stage_entity_extraction_sec', 'p(95)'), 3)}`);
  lines.push(`ticket_creation_sec p95: ${fmt(metricValue(data, 'stage_ticket_creation_sec', 'p(95)'), 3)}`);
  lines.push(`notification_sec p95: ${fmt(metricValue(data, 'stage_notification_sec', 'p(95)'), 3)}`);
  lines.push(`http_5xx rate: ${fmt(metricValue(data, 'http_5xx_rate', 'rate'), 4)}`);
  lines.push(`payload_incomplete rate: ${fmt(metricValue(data, 'payload_incomplete_rate', 'rate'), 4)}`);
  lines.push(`result_completed count: ${fmt(metricValue(data, 'result_completed_count', 'count'), 0)}`);
  lines.push(`result_spam_blocked count: ${fmt(metricValue(data, 'result_spam_blocked_count', 'count'), 0)}`);
  lines.push(`result_review_required count: ${fmt(metricValue(data, 'result_review_required_count', 'count'), 0)}`);
  lines.push(`dropped_iterations count: ${fmt(metricValue(data, 'dropped_iterations', 'count'), 0)}`);
  return `${lines.join('\n')}\n`;
}

const baseUrl = env('BASE_URL', 'http://localhost:8000').replace(/\/+$/, '');
const thinkTimeSec = numEnv('THINK_TIME_SEC', 0.2);
const workloadModel = env('WORKLOAD_MODEL', 'closed').toLowerCase();
const audioFixtures = parseAudioFixtures();
const scenarioConfig = buildScenario();

const orchestratorTotal = new Trend('orchestrator_total_sec');
const transcriptionStage = new Trend('stage_transcription_sec');
const routingStage = new Trend('stage_routing_sec');
const entityStage = new Trend('stage_entity_extraction_sec');
const ticketStage = new Trend('stage_ticket_creation_sec');
const notificationStage = new Trend('stage_notification_sec');
const payloadIncompleteRate = new Rate('payload_incomplete_rate');
const http5xxRate = new Rate('http_5xx_rate');
const resultCompletedCount = new Counter('result_completed_count');
const resultSpamBlockedCount = new Counter('result_spam_blocked_count');
const resultReviewRequiredCount = new Counter('result_review_required_count');

export const options = {
  scenarios: {
    [scenarioConfig.name]: scenarioConfig.scenario,
  },
  thresholds: {
    http_req_failed: [`rate<${env('MAX_HTTP_ERROR_RATE', '0.05')}`],
    http_req_duration: [
      `p(95)<${env('HTTP_P95_MS', '300000')}`,
      `p(99)<${env('HTTP_P99_MS', '600000')}`,
    ],
    checks: [`rate>${env('MIN_CHECK_RATE', '0.95')}`],
    payload_incomplete_rate: [`rate<${env('MAX_INCOMPLETE_RATE', '0.02')}`],
    http_5xx_rate: [`rate<${env('MAX_HTTP_5XX_RATE', '0.02')}`],
    orchestrator_total_sec: [
      `p(95)<${env('PIPELINE_P95_SEC', '300')}`,
      `p(99)<${env('PIPELINE_P99_SEC', '600')}`,
    ],
    dropped_iterations: [`count<=${env('MAX_DROPPED_ITERATIONS', '0')}`],
  },
};

export function setup() {
  const tokenFromEnv = env('TOKEN');
  if (tokenFromEnv) {
    return { token: tokenFromEnv };
  }

  const username = requireEnv('USERNAME');
  const password = requireEnv('PASSWORD');
  const response = http.post(
    `${baseUrl}/api/v1/auth/login`,
    JSON.stringify({ username, password }),
    { headers: { 'Content-Type': 'application/json', Accept: 'application/json' } },
  );

  check(response, {
    'login status is 200': (res) => res.status === 200,
  });

  const payload = safeJson(response);
  if (!payload || !payload.token) {
    throw new Error(`Login did not return token. Status=${response.status} body=${response.body}`);
  }

  return { token: String(payload.token) };
}

export default function (data) {
  const file = audioFixtures[__ITER % audioFixtures.length];
  const response = http.post(
    `${baseUrl}/api/v1/process-call`,
    {
      audio: http.file(file.bytes, file.name),
    },
    {
      headers: {
        Authorization: `Bearer ${data.token}`,
        Accept: 'application/json',
      },
      tags: {
        endpoint: 'process_call',
        audio_name: file.name,
        workload_model: workloadModel,
      },
      timeout: env('REQUEST_TIMEOUT', '3600s'),
    },
  );

  const payload = safeJson(response);
  const processingTime = payload && typeof payload.processing_time === 'object' ? payload.processing_time : null;
  const status = payload && typeof payload.status === 'string' ? payload.status : '';

  http5xxRate.add(response.status >= 500 ? 1 : 0);
  payloadIncompleteRate.add(
    payload && payload.transcript && payload.routing ? 0 : 1,
    { status: response.status, audio_name: file.name },
  );

  const totalTime = payload ? safeNumber(payload.total_time) : null;
  if (totalTime !== null) {
    orchestratorTotal.add(totalTime, { status, audio_name: file.name });
  }
  if (processingTime) {
    const transcription = safeNumber(processingTime.transcription);
    const routing = safeNumber(processingTime.routing);
    const entityExtraction = safeNumber(processingTime.entity_extraction);
    const ticketCreation = safeNumber(processingTime.ticket_creation);
    const notification = safeNumber(processingTime.notification);
    if (transcription !== null) {
      transcriptionStage.add(transcription, { status, audio_name: file.name });
    }
    if (routing !== null) {
      routingStage.add(routing, { status, audio_name: file.name });
    }
    if (entityExtraction !== null) {
      entityStage.add(entityExtraction, { status, audio_name: file.name });
    }
    if (ticketCreation !== null) {
      ticketStage.add(ticketCreation, { status, audio_name: file.name });
    }
    if (notification !== null) {
      notificationStage.add(notification, { status, audio_name: file.name });
    }
  }

  if (status === 'completed') {
    resultCompletedCount.add(1);
  } else if (status === 'spam_blocked') {
    resultSpamBlockedCount.add(1);
  } else if (status === 'awaiting_spam_review') {
    resultReviewRequiredCount.add(1);
  }

  check(response, {
    'process-call status is 200': (res) => res.status === 200,
    'response contains transcript': () => !!(payload && payload.transcript),
    'response contains routing': () => !!(payload && payload.routing),
    'response contains status': () => !!status,
    'response contains total_time': () => totalTime !== null,
  });

  if (workloadModel === 'closed' && thinkTimeSec > 0) {
    sleep(thinkTimeSec);
  }
}

export function handleSummary(data) {
  const jsonPath = env('SUMMARY_JSON', 'load_test_summary.json');
  const textPath = env('SUMMARY_TEXT', 'load_test_summary.txt');
  const text = buildTextSummary(data);

  return {
    stdout: text,
    [jsonPath]: JSON.stringify(data, null, 2),
    [textPath]: text,
  };
}
