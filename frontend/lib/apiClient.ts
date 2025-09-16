export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

type Json = Record<string, unknown> | Array<unknown> | string | number | boolean | null;

export interface ApiErrorEnvelope {
  error: { code: string; message: string; violations?: Array<{ field?: string; reason?: string }>; };
}

export async function apiFetch<T = unknown>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), init?.timeoutMs ?? 30000);

  const res = await fetch(`${API_BASE}${path}`,
    {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers || {}),
      },
      signal: controller.signal,
      cache: 'no-store',
    },
  ).finally(() => clearTimeout(timeout));

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

