import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  SurveillanceService,
  HealthResponse,
  ImageDetectionResponse,
  VideoStatusResponse,
  LiveFrameResponse,
  AlertEvent,
  TrackedEntity
} from './services/surveillance.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent implements OnInit, OnDestroy {
  activeTab: 'video' | 'image' | 'live' = 'video';
  currentTheme: 'dark' | 'light' = 'dark';

  // System status
  backendOnline = false;
  healthInfo: HealthResponse | null = null;
  currentTime = new Date();
  private timeInterval: any;

  // Settings
  soundAlertsEnabled = true;

  // --- Image Tab State ---
  imageFile: File | null = null;
  imagePreview: string | null = null;
  imageConfidence = 0.25;
  imageApplyClahe = false;
  imageLoading = false;
  imageResult: ImageDetectionResponse | null = null;
  imageError: string | null = null;

  // --- Video Tab State ---
  videoFile: File | null = null;
  videoConfidence = 0.15;
  videoLoading = false;
  videoTaskId: string | null = null;
  videoStatus: VideoStatusResponse | null = null;
  videoResultUrl: string | null = null;
  videoError: string | null = null;
  selectedAlert: AlertEvent | null = null;
  selectedEntity: TrackedEntity | null = null;
  entityFilter: 'all' | 'alerts' | 'fusion' = 'all';
  currentPlaybackSpeed = 1.0;
  private videoPollingInterval: any;

  @ViewChild('surveillanceVideo') surveillanceVideo?: ElementRef<HTMLVideoElement>;

  // --- Live Camera Tab State ---
  liveActive = false;
  liveConfidence = 0.55;
  liveTelemetry: any = null;
  liveError: string | null = null;
  private liveWs: WebSocket | null = null;
  private liveStream: MediaStream | null = null;
  private liveFrameInterval: any;

  @ViewChild('webcamVideo') webcamVideo!: ElementRef<HTMLVideoElement>;
  @ViewChild('captureCanvas') captureCanvas!: ElementRef<HTMLCanvasElement>;
  liveAnnotatedImage: string | null = null;

  private audioCtx: AudioContext | null = null;

  constructor(private surveillanceService: SurveillanceService) {}

  ngOnInit() {
    const savedTheme = localStorage.getItem('aegis_theme') as 'dark' | 'light' | null;
    if (savedTheme) {
      this.currentTheme = savedTheme;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      this.currentTheme = 'light';
    }

    this.checkHealth();
    this.timeInterval = setInterval(() => {
      this.currentTime = new Date();
    }, 1000);
  }

  ngOnDestroy() {
    if (this.timeInterval) clearInterval(this.timeInterval);
    if (this.videoPollingInterval) clearInterval(this.videoPollingInterval);
    this.stopLiveCamera();
    if (this.audioCtx) {
      this.audioCtx.close();
    }
  }

  toggleTheme() {
    this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('aegis_theme', this.currentTheme);
  }

  checkHealth() {
    this.surveillanceService.checkHealth().subscribe({
      next: (res) => {
        this.backendOnline = true;
        this.healthInfo = res;
      },
      error: () => {
        this.backendOnline = false;
        this.healthInfo = null;
      }
    });
  }

  setTab(tab: 'video' | 'image' | 'live') {
    if (this.activeTab === 'live' && tab !== 'live') {
      this.stopLiveCamera();
    }
    this.activeTab = tab;
  }

  playSubtleBeep(freq = 640, duration = 0.12) {
    if (!this.soundAlertsEnabled) return;
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
      gain.gain.setValueAtTime(0.08, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start();
      osc.stop(this.audioCtx.currentTime + duration);
    } catch {}
  }

  // ================= IMAGE TAB =================
  onImageFileSelected(event: any) {
    const file = event.target.files?.[0];
    if (file) {
      this.imageFile = file;
      this.imageResult = null;
      this.imageError = null;

      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result as string;
      };
      reader.readAsDataURL(file);
    }
  }

  onImageDrop(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer?.files?.length) {
      const file = event.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        this.imageFile = file;
        this.imageResult = null;
        this.imageError = null;

        const reader = new FileReader();
        reader.onload = () => {
          this.imagePreview = reader.result as string;
        };
        reader.readAsDataURL(file);
      }
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
  }

  processImage() {
    if (!this.imageFile) return;

    this.imageLoading = true;
    this.imageError = null;

    this.surveillanceService.detectImage(this.imageFile, this.imageConfidence, this.imageApplyClahe)
      .subscribe({
        next: (res) => {
          this.imageResult = res;
          this.imageLoading = false;
          if (res.person_count > 0) {
            this.playSubtleBeep(720, 0.15);
          }
        },
        error: (err) => {
          this.imageError = err.error?.detail || err.message || 'Unable to process image.';
          this.imageLoading = false;
        }
      });
  }

  // ================= VIDEO TAB =================
  onVideoFileSelected(event: any) {
    const file = event.target.files?.[0];
    if (file) {
      this.videoFile = file;
      this.videoStatus = null;
      this.videoResultUrl = null;
      this.videoError = null;
      this.selectedAlert = null;
      this.selectedEntity = null;
    }
  }

  processVideo() {
    if (!this.videoFile) return;

    this.videoLoading = true;
    this.videoError = null;
    this.videoStatus = null;
    this.videoResultUrl = null;
    this.selectedAlert = null;
    this.selectedEntity = null;

    this.surveillanceService.startVideoProcessing(this.videoFile, this.videoConfidence).subscribe({
      next: (res) => {
        this.videoTaskId = res.task_id;
        this.pollVideoStatus(res.task_id);
      },
      error: (err) => {
        this.videoError = err.error?.detail || err.message || 'Unable to start video processing.';
        this.videoLoading = false;
      }
    });
  }

  private pollVideoStatus(taskId: string) {
    if (this.videoPollingInterval) {
      clearInterval(this.videoPollingInterval);
    }

    this.videoPollingInterval = setInterval(() => {
      this.surveillanceService.getVideoStatus(taskId).subscribe({
        next: (status) => {
          this.videoStatus = status;
          if (status.status === 'completed') {
            clearInterval(this.videoPollingInterval);
            this.videoLoading = false;
            this.videoResultUrl = this.surveillanceService.getVideoResultUrl(taskId);
            if (status.alert_count > 0) {
              this.playSubtleBeep(520, 0.25);
            }
          } else if (status.status === 'error') {
            clearInterval(this.videoPollingInterval);
            this.videoLoading = false;
            this.videoError = status.error_message || 'Video analysis encountered an error.';
          }
        },
        error: () => {
          clearInterval(this.videoPollingInterval);
          this.videoLoading = false;
          this.videoError = 'Failed to fetch video processing status.';
        }
      });
    }, 1200);
  }

  jumpToAlert(alert: AlertEvent) {
    this.selectedAlert = alert;
    this.playSubtleBeep(680, 0.1);
    const video = this.surveillanceVideo?.nativeElement;
    if (video) {
      video.currentTime = Math.max(0, alert.start_time_sec - 0.5);
      video.play().catch(() => {});
    }
  }

  jumpToEntity(entity: TrackedEntity) {
    this.selectedEntity = entity;
    this.playSubtleBeep(640, 0.1);
    const video = this.surveillanceVideo?.nativeElement;
    if (video) {
      video.currentTime = Math.max(0, entity.first_seen_sec);
      video.play().catch(() => {});
    }
  }

  setPlaybackSpeed(speed: number) {
    this.currentPlaybackSpeed = speed;
    const video = this.surveillanceVideo?.nativeElement;
    if (video) {
      video.playbackRate = speed;
    }
  }

  get filteredEntities(): TrackedEntity[] {
    const list = this.videoStatus?.tracked_entities || [];
    if (this.entityFilter === 'alerts') {
      return list.filter(e => e.confirmed_alert);
    }
    if (this.entityFilter === 'fusion') {
      return list.filter(e => e.camera_tags.includes('A+B'));
    }
    return list;
  }

  // ================= LIVE CAMERA TAB =================
  async toggleLiveCamera() {
    if (this.liveActive) {
      this.stopLiveCamera();
    } else {
      await this.startLiveCamera();
    }
  }

  async startLiveCamera() {
    this.liveError = null;
    try {
      this.liveStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });

      if (this.webcamVideo && this.webcamVideo.nativeElement) {
        this.webcamVideo.nativeElement.srcObject = this.liveStream;
        await this.webcamVideo.nativeElement.play();
      }

      this.initWebSocket();
      this.liveActive = true;
    } catch (err: any) {
      this.liveError = 'Camera access was denied or is unavailable: ' + (err.message || err);
      this.stopLiveCamera();
    }
  }

  private initWebSocket() {
    try {
      this.liveWs = this.surveillanceService.createWebSocket();

      this.liveWs.onopen = () => {
        this.startStreamingFrames();
      };

      this.liveWs.onmessage = (event) => {
        try {
          const res: LiveFrameResponse = JSON.parse(event.data);
          this.liveAnnotatedImage = res.image;
          this.liveTelemetry = res.telemetry;
          if (res.telemetry?.threat_count > 0) {
            this.playSubtleBeep(750, 0.08);
          }
        } catch (e) {
          console.error('WebSocket parse error', e);
        }
      };

      this.liveWs.onerror = () => {
        this.liveError = 'Live connection error to backend.';
      };

      this.liveWs.onclose = () => {
        if (this.liveActive) {
          this.stopLiveCamera();
        }
      };
    } catch (e: any) {
      this.liveError = 'Failed to connect: ' + e.message;
    }
  }

  private startStreamingFrames() {
    const canvas = this.captureCanvas?.nativeElement;
    const video = this.webcamVideo?.nativeElement;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    this.liveFrameInterval = setInterval(() => {
      if (!this.liveActive || !this.liveWs || this.liveWs.readyState !== WebSocket.OPEN) {
        return;
      }

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = 640;
        canvas.height = Math.round(640 * (video.videoHeight / video.videoWidth));
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.65);

        this.liveWs.send(JSON.stringify({
          image: dataUrl,
          conf: this.liveConfidence
        }));
      }
    }, 150);
  }

  stopLiveCamera() {
    this.liveActive = false;
    if (this.liveFrameInterval) clearInterval(this.liveFrameInterval);
    if (this.liveWs) {
      this.liveWs.close();
      this.liveWs = null;
    }
    if (this.liveStream) {
      this.liveStream.getTracks().forEach(track => track.stop());
      this.liveStream = null;
    }
    this.liveAnnotatedImage = null;
    this.liveTelemetry = null;
  }
}
