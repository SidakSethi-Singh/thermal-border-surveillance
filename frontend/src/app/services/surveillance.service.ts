import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface HealthResponse {
  status: string;
  thermal_model_exists: boolean;
  live_model_exists: boolean;
  service: string;
}

export interface DetectionItem {
  class: string;
  confidence: number;
  confidence_pct: string;
  bbox: number[];
}

export interface ImageDetectionResponse {
  annotated_image: string;
  person_count: number;
  person_detections: DetectionItem[];
  other_detections: DetectionItem[];
  total_detections: number;
}

export interface VideoQueueResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface AlertEvent {
  alert_id: number;
  track_id: number;
  behavior: string;
  suspicion: number;
  cam_tag: string;
  start_frame: number;
  start_time_sec: number;
  start_time_formatted: string;
  duration_frames: number;
  severity: 'CRITICAL' | 'ELEVATED';
}

export interface TrackedEntity {
  track_id: number;
  dominant_behavior: string;
  max_suspicion: number;
  max_suspicion_pct: string;
  confirmed_alert: boolean;
  alert_frames: number;
  camera_tags: string[];
  first_seen_sec: number;
  first_seen_formatted: string;
  last_seen_sec: number;
  last_seen_formatted: string;
  duration_sec: number;
}

export interface ClassificationBreakdown {
  crouching: number;
  group: number;
  walking: number;
  stationary: number;
  non_threat_objects: number;
  fusion_overlap: number;
}

export interface VideoStatusResponse {
  task_id: string;
  status: 'queued' | 'processing' | 'completed' | 'error';
  progress: number;
  alert_count: number;
  tracks_count: number;
  total_frames: number;
  processed_frames: number;
  duration_sec?: number;
  fps?: number;
  classification_breakdown?: ClassificationBreakdown;
  confirmed_alerts?: AlertEvent[];
  tracked_entities?: TrackedEntity[];
  non_human_summary?: Record<string, number>;
  error_message?: string;
  result_url?: string;
}

export interface LiveTrackItem {
  id: number;
  status: 'threat' | 'normal' | 'safe';
  speed: number;
  aspect: number;
  grouped: boolean;
  bbox: number[];
}

export interface LiveFrameResponse {
  image: string;
  telemetry: {
    active_persons: number;
    threat_count: number;
    other_objects: number;
    tracks: LiveTrackItem[];
  };
}

@Injectable({
  providedIn: 'root'
})
export class SurveillanceService {
  private readonly baseUrl = 'http://localhost:8000';
  private readonly wsUrl = 'ws://localhost:8000/api/ws/live';

  constructor(private http: HttpClient) {}

  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/api/health`);
  }

  detectImage(file: File, conf: number = 0.25, applyClahe: boolean = false): Observable<ImageDetectionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    let params = new HttpParams()
      .set('conf', conf.toString())
      .set('apply_clahe', applyClahe.toString());

    return this.http.post<ImageDetectionResponse>(`${this.baseUrl}/api/detect/image`, formData, { params });
  }

  startVideoProcessing(file: File, conf: number = 0.15): Observable<VideoQueueResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const params = new HttpParams().set('conf', conf.toString());
    return this.http.post<VideoQueueResponse>(`${this.baseUrl}/api/detect/video`, formData, { params });
  }

  getVideoStatus(taskId: string): Observable<VideoStatusResponse> {
    return this.http.get<VideoStatusResponse>(`${this.baseUrl}/api/detect/video/status/${taskId}`);
  }

  getVideoResultUrl(taskId: string): string {
    return `${this.baseUrl}/api/detect/video/result/${taskId}`;
  }

  createWebSocket(): WebSocket {
    return new WebSocket(this.wsUrl);
  }
}
