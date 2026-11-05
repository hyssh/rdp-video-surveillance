export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export interface VideoItem {
  id: number;
  filename: string | null;
  security_level: string | null;
  risk_score: number | null;
  created_at: string | null;
}
export interface VideoList { items: VideoItem[]; total: number; limit: number; offset: number }
export interface VideoDetail extends VideoItem {
  summary: string | null;
  frame_count: number;
  frames_described: number;
}
export interface Stats {
  videos_total: number;
  frames_total: number;
  frames_described: number;
  high_risk_videos: number;
  risk_distribution: Record<RiskLevel, number>;
  action_types: { action_type: string; count: number }[];
  recent_videos: VideoItem[];
}
export interface TimelinePoint {
  frame_id: number;
  t: number;
  risk_score: number | null;
  risk_level: string | null;
  action_type: string | null;
}
export interface TimelineEvent {
  frame_id: number;
  t: number;
  risk_level: string | null;
  action_type: string | null;
  description: string | null;
}
export interface Timeline {
  video_id: number;
  filename: string | null;
  duration_s: number | null;
  interval_s: number | null;
  summary: string | null;
  security_level: string | null;
  risk_score: number | null;
  points: TimelinePoint[];
  events: TimelineEvent[];
}
export interface FrameDetail {
  id: number;
  video_id: number;
  frame_timestamp: number | null;
  image_description: string | null;
  risk_score: number | null;
  risk_level: string | null;
  action_type: string | null;
  image_url: string;
  frame_width: number | null;
  frame_height: number | null;
}
export interface UiElement {
  type: string;
  content: string | null;
  interactivity: boolean | null;
  bbox: [number, number, number, number];
}
export interface FrameElements { frame_id: number; width: number | null; height: number | null; elements: UiElement[] }
export interface AskMatch {
  frame_id: number;
  frame_timestamp: number | null;
  image_description: string | null;
  similarity: number;
  url: string | null;
}
export interface AskResponse {
  video_id: number;
  question: string;
  answer: string | null;
  matches: AskMatch[];
  note: string | null;
}
export interface JobStatus {
  job_id: string;
  status: 'queued' | 'extracting' | 'analyzing' | 'summarizing' | 'done' | 'failed';
  stage: string | null;
  frames_done: number;
  frames_total: number;
  frames_failed: number;
  video_id: number | null;
  error: string | null;
  warnings: string[];
}
