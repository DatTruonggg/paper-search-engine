export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8001";

type Json = Record<string, unknown> | Array<unknown> | string | number | boolean | null;

export interface ApiErrorEnvelope {
  error: { code: string; message: string; violations?: Array<{ field?: string; reason?: string }>; };
}

export async function apiFetch<T = unknown>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  // Support external AbortSignal: if provided, we won't create our own controller/timeout
  const externalSignal = init?.signal;
  const controller = externalSignal ? undefined : new AbortController();
  const timeoutMs = init?.timeoutMs ?? 30000;
  const timeout = !externalSignal && timeoutMs > 0 ? setTimeout(() => (controller as AbortController).abort(), timeoutMs) : undefined;

  const res = await fetch(`${API_BASE}${path}`,
    {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers || {}),
      },
      signal: externalSignal ?? controller?.signal,
      cache: 'no-store',
    },
  ).finally(() => {
    if (timeout) clearTimeout(timeout);
  });

  if (!res.ok) {
    let body: ApiErrorEnvelope | undefined;
    try { body = await res.json(); } catch {}
    const message = body?.error?.message || `HTTP ${res.status}`;
    const error = new Error(message) as Error & { code?: string; status?: number; violations?: unknown };
    error.code = body?.error?.code;
    error.status = res.status;
    error.violations = body?.error?.violations;
    throw error;
  }

  if (res.status === 204) return undefined as unknown as T;
  return res.json() as Promise<T>;
}

