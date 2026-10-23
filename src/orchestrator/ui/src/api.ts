import type { AskResponse, FrameDetail, FrameElements, JobStatus, Stats, Timeline, VideoDetail, VideoList } from './types';

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

export class ApiError extends Error {
  error: string;
  stage?: string;
  status: number;
  constructor(error: string, detail: string, status = 0, stage?: string) {
    super(detail);
    this.error = error;
    this.stage = stage;
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(API_BASE + path, init);
  } catch {
    throw new ApiError('network_error', 'Could not reach the server. Is the API running?');
  }
  const text = await resp.text();
  let body: any = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }
  if (resp.ok) return body as T;
  if (body && typeof body.error === 'string') throw new ApiError(body.error, body.detail ?? '', resp.status, body.stage);
  if (body && typeof body.detail === 'string') throw new ApiError(`http_${resp.status}`, body.detail, resp.status);
  throw new ApiError(`http_${resp.status}`, resp.statusText || 'Request failed', resp.status);
}

export const imageUrl = (frameId: number) => `${API_BASE}/frames/${frameId}/image`;
export const clipUrl = (videoId: number, t: number) => `${API_BASE}/videos/${videoId}/clip?t=${t}&before=3&after=3`;

export const api = {
  stats: () => request<Stats>('/stats'),
  videos: () => request<VideoList>('/videos?limit=200'),
  video: (id: number) => request<VideoDetail>(`/videos/${id}`),
  timeline: (id: number) => request<Timeline>(`/videos/${id}/timeline`),
  frame: (id: number) => request<FrameDetail>(`/frames/${id}`),
  elements: (id: number) => request<FrameElements>(`/frames/${id}/elements`),
  job: (id: string) => request<JobStatus>(`/jobs/${id}`),
  ask: (id: number, question: string, top_k = 5) =>
    request<AskResponse>(`/videos/${id}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k }),
    }),
  ingest: (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return request<{ job_id: string; status: string; status_url: string }>('/v2/ingest-video/', { method: 'POST', body: fd });
  },
  async clip(videoId: number, t: number): Promise<string> {
    let resp: Response;
    try { resp = await fetch(clipUrl(videoId, t)); } catch {
      throw new ApiError('network_error', 'Could not reach the server.');
    }
    if (!resp.ok) {
      let b: any = null;
      try { b = await resp.json(); } catch { /* ignore */ }
      throw new ApiError(b?.error ?? `http_${resp.status}`, b?.detail ?? 'Clip unavailable', resp.status, b?.stage);
    }
    return URL.createObjectURL(await resp.blob());
  },
};

const FRIENDLY: Record<string, string> = {
  clip_unavailable: 'The source video isn’t available for this session, so a clip can’t be played. Use the frame filmstrip instead.',
  network_error: 'Can’t reach the analysis server. Check that the backend is running.',
  video_not_found: 'That session doesn’t exist (it may have been removed).',
  frame_not_found: 'That frame couldn’t be found.',
  missing_config: 'The server is missing configuration needed for this action.',
  upstream_openai_error: 'The AI service returned an error. Please retry in a moment.',
  upstream_omniparser_error: 'The screen parser service returned an error. Please retry in a moment.',
};

export function friendlyError(e: unknown): { title: string; detail?: string; stage?: string } {
  if (e instanceof ApiError) {
    return { title: FRIENDLY[e.error] ?? 'Something went wrong', detail: e.message && !FRIENDLY[e.error] ? e.message : e.message, stage: e.stage };
  }
  return { title: 'Something went wrong', detail: e instanceof Error ? e.message : String(e) };
}
