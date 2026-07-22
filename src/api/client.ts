import Constants from 'expo-constants';
import type { PredictRequest, PredictResponse } from './types';

const API_BASE_URL: string =
  (Constants.expoConfig?.extra?.apiBaseUrl as string | undefined) ?? 'http://localhost:8000';

// Render's free tier sleeps after 15 min idle; first request after that can
// take 30-60s to cold-start, so this timeout is deliberately generous.
const REQUEST_TIMEOUT_MS = 45000;

async function fetchWithTimeout(url: string, options: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function getAirports(): Promise<string[]> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/airports`, { method: 'GET' });
  if (!res.ok) throw new Error(`Failed to load airports (${res.status})`);
  return res.json();
}

export async function getAirlines(): Promise<string[]> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/airlines`, { method: 'GET' });
  if (!res.ok) throw new Error(`Failed to load airlines (${res.status})`);
  return res.json();
}

export async function predictDelay(req: PredictRequest): Promise<PredictResponse> {
  const res = await fetchWithTimeout(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Prediction failed (${res.status}): ${body}`);
  }
  return res.json();
}
