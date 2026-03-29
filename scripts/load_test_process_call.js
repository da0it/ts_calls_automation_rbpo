import http from 'k6/http';
import { check, sleep } from 'k6';

function requireEnv(name) {
  const value = (__ENV[name] || '').trim();
  if (!value) {
    throw new Error(`Missing required env: ${name}`);
  }
  return value;
}

function parseAudioFixtures() {
  const raw = (__ENV.AUDIO_FILES || '').trim();
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

const baseUrl = (__ENV.BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');
const thinkTimeSec = Number(__ENV.THINK_TIME_SEC || '0.2');
const audioFixtures = parseAudioFixtures();

export const options = {
  scenarios: {
    process_call_load: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: __ENV.RAMP_UP_1 || '2m', target: Number(__ENV.TARGET_VUS_1 || '20') },
        { duration: __ENV.RAMP_UP_2 || '3m', target: Number(__ENV.TARGET_VUS_2 || '60') },
        { duration: __ENV.SOAK || '5m', target: Number(__ENV.TARGET_VUS_2 || '60') },
        { duration: __ENV.RAMP_DOWN || '1m', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.03'],
    http_req_duration: ['p(95)<2000', 'avg<1500'],
    checks: ['rate>0.97'],
  },
};

export function setup() {
  const tokenFromEnv = (__ENV.TOKEN || '').trim();
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
      timeout: __ENV.REQUEST_TIMEOUT || '3600s',
    },
  );
  const payload = safeJson(response);

  check(response, {
    'process-call status is 200': (res) => res.status === 200,
    'response contains transcript': () => !!(payload && payload.transcript),
    'response contains routing': () => !!(payload && payload.routing),
  });

  sleep(thinkTimeSec);
}
