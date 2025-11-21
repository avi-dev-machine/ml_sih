import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

class Timer:
    def __init__(self):
        self._start_time: Optional[float] = None
        self._total_time: float = 0.0

    def start(self):
        if self._start_time is None:
            self._start_time = time.time()

    def end(self):
        if self._start_time is not None:
            self._total_time += time.time() - self._start_time
            self._start_time = None

    def get_current_time(self) -> float:
        if self._start_time is None:
            return self._total_time
        return self._total_time + (time.time() - self._start_time)

    def convert_time(self, seconds: float):
        # returns HH:MM:SS.mmm as string
        ms = int((seconds - int(seconds)) * 1000)
        s = int(seconds) % 60
        m = (int(seconds) // 60) % 60
        h = int(seconds) // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class PlankHold:
    hold_number: int
    start_time: Optional[float]
    end_time: Optional[float]
    duration: float
    hip_angle: float
    alignment: float
    hip_sag: float
    form_quality: float


class PlankAnalyzer:
    """Simple analyzer for plank holds.

    Heuristics:
    - A hold starts when hip_angle > HOLD_ANGLE and enough keypoints exist.
    - A hold ends when hip_angle drops below RELEASE_ANGLE.
    - Form quality is computed from hip alignment and hip sag.
    """

    def __init__(self, hold_angle=160, release_angle=150, history_length=30):
        self.HOLD_ANGLE = hold_angle
        self.RELEASE_ANGLE = release_angle
        # Debounce frames required to start/end a hold
        self.HOLD_FRAMES = 5
        self.RELEASE_FRAMES = 5

        # Counters for consecutive frames meeting condition
        self._hold_candidate_frames = 0
        self._release_candidate_frames = 0

        self.holds: List[PlankHold] = []
        self.stage = 'not_holding'
        self.current_start = None
        self.current_min_sag = float('inf')
        self.current_max_sag = 0
        self.current_hip_angles = []
        self.counter = 0

        # rolling history if needed
        self.hip_angle_history = []

        # ----- Pose-based plank detection attributes (from user's algorithm) -----
        self.ang1_tracker: List[float] = []
        self.ang4_tracker: List[float] = []
        self.plank_counter: int = 0
        self.timer = Timer() if 'Timer' in globals() else None
        self.start_time: Optional[float] = None
        self.total_time: float = 0.0

    # ----------------- Pose-angle helpers -----------------
    def _angle_between_three(self, a, b, c):
        # compute angle at b between ba and bc
        try:
            a = np.array(a); b = np.array(b); c = np.array(c)
            v1 = a - b
            v2 = c - b
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return None
            u1 = v1 / n1
            u2 = v2 / n2
            dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
            ang = float(np.degrees(np.arccos(dot)))
            return ang
        except Exception:
            return None

    def _angle_line(self, p1, p2):
        try:
            p1 = np.array(p1); p2 = np.array(p2)
            radians = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
            deg = abs(np.degrees(radians))
            if deg > 180:
                deg = 360 - deg
            return deg
        except Exception:
            return None
    def process_pose_plank(self, keypoints, fps: int = 30):
        """Process a frame's keypoints using the user's Plank logic.

        keypoints: numpy array of shape (N,2) with indices matching pushup model.
        Returns a dict with status and timer info for display.
        """
        ang1 = ang2 = ang3 = ang4 = None

        # indices assumed: left shoulder=5, left elbow=7, left wrist=9, left hip=11, left ankle=15
        # prefer left side, else right side (6,8,10,12,16)
        try:
            p = keypoints
            # ang1: shoulder-elbow-wrist
            if len(p) > 9 and not np.any(np.isnan(p[[5,7,9]])):
                ang1 = self._angle_between_three(p[5], p[7], p[9])
            elif len(p) > 10 and not np.any(np.isnan(p[[6,8,10]])):
                ang1 = self._angle_between_three(p[6], p[8], p[10])

            # ang2: shoulder-hip-ankle
            if len(p) > 11 and not np.any(np.isnan(p[[5,11,15]])):
                ang2 = self._angle_between_three(p[5], p[11], p[15])
            elif len(p) > 12 and not np.any(np.isnan(p[[6,12,16]])):
                ang2 = self._angle_between_three(p[6], p[12], p[16])

            # ang3: angle of line shoulder-ankle or hip-ankle
            left_shoulder_ankle = len(p) > 15 and not np.any(np.isnan(p[[5,15]]))
            right_shoulder_ankle = len(p) > 16 and not np.any(np.isnan(p[[6,16]]))
            left_hip_ankle = len(p) > 15 and not np.any(np.isnan(p[[11,15]]))
            right_hip_ankle = len(p) > 16 and not np.any(np.isnan(p[[12,16]]))
            if left_shoulder_ankle or right_shoulder_ankle:
                shoulder_idx = 5 if left_shoulder_ankle else 6
                ankle_idx = 15 if left_shoulder_ankle else 16
                ang3 = self._angle_line(p[shoulder_idx], p[ankle_idx])
            elif left_hip_ankle or right_hip_ankle:
                hip_idx = 11 if left_hip_ankle else 12
                ankle_idx = 15 if left_hip_ankle else 16
                ang3 = self._angle_line(p[hip_idx], p[ankle_idx])

            # ang4: angle of line elbow-wrist
            left_elbow_wrist = len(p) > 9 and not np.any(np.isnan(p[[7,9]]))
            right_elbow_wrist = len(p) > 10 and not np.any(np.isnan(p[[8,10]]))
            if left_elbow_wrist or right_elbow_wrist:
                elbow_idx = 7 if left_elbow_wrist else 8
                wrist_idx = 9 if left_elbow_wrist else 10
                ang4 = self._angle_line(p[elbow_idx], p[wrist_idx])
        except Exception:
            pass

        # follow user's logic
        if ang3 is not None and ((0 <= ang3 <= 50) or (130 <= ang3 <= 180)):
            if (ang1 is not None or ang2 is not None) and ang4 is not None:
                if (ang2 is not None and (160 <= ang2 <= 180)) or (ang2 is not None and (0 <= ang2 <= 20)):
                    self.plank_counter += 1
                    if ang1 is not None:
                        self.ang1_tracker.append(ang1)
                    else:
                        # append last value or zero to keep trackers aligned
                        self.ang1_tracker.append(self.ang1_tracker[-1] if self.ang1_tracker else 0.0)
                    self.ang4_tracker.append(ang4)

        # evaluate when we have a window of 24
        if self.plank_counter >= 24 and len(self.ang1_tracker) >= 24 and len(self.ang4_tracker) >= 24:
            ang1_diff1 = abs(self.ang1_tracker[0] - self.ang1_tracker[12])
            ang1_diff2 = abs(self.ang1_tracker[12] - self.ang1_tracker[23])
            ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2.0
            ang4_mean = sum(self.ang4_tracker[:24]) / 24.0
            # pop oldest to implement sliding window
            del self.ang1_tracker[0]
            del self.ang4_tracker[0]
            if ang1_diff_mean < 5 and not (75 <= ang4_mean <= 105):
                # considered plank -> start timer
                if self.timer is None:
                    self.timer = Timer()
                if self.start_time is None:
                    self.timer.start()
                self.start_time = self.timer._start_time
            else:
                # not plank -> stop timer if running
                if self.start_time is not None and self.timer is not None:
                    self.timer.end()
                    self.total_time = self.timer.get_current_time()
                self.start_time = None

        # build display info
        timer_time = self.timer.get_current_time() if self.timer is not None else 0.0
        return {
            'stage': 'plank' if self.start_time is not None else 'not_plank',
            'counter': self.plank_counter,
            'timer': timer_time,
            'total_time': self.total_time,
            'ang1': ang1,
            'ang2': ang2,
            'ang3': ang3,
            'ang4': ang4,
        }

    def _calc_alignment_score(self, shoulder, hip, ankle):
        # Very simple geometric alignment: distance of hip from line shoulder-ankle
        try:
            shoulder = np.array(shoulder, dtype=np.float32)
            hip = np.array(hip, dtype=np.float32)
            ankle = np.array(ankle, dtype=np.float32)
            v = ankle - shoulder
            v_len = np.linalg.norm(v)
            if v_len < 1e-6:
                return 100.0
            v_unit = v / v_len
            hip_v = hip - shoulder
            perp = abs(hip_v[0] * (-v_unit[1]) + hip_v[1] * v_unit[0])
            # normalize assuming 50 px is bad
            score = max(0.0, 100.0 - (perp / 50.0) * 100.0)
            return float(min(100.0, score))
        except Exception:
            return 100.0

    def _calc_form_quality(self, hip_angle, alignment, hip_sag):
        # Weighted: alignment 50%, hip_angle closeness-to-180 30%, hip_sag 20%
        angle_score = max(0.0, min(100.0, (hip_angle / 180.0) * 100.0))
        hip_sag_score = max(0.0, 100.0 - (hip_sag / 30.0) * 100.0)
        quality = angle_score * 0.30 + alignment * 0.50 + hip_sag_score * 0.20
        return float(max(0.0, min(100.0, quality)))

    def update(self, hip_angle, shoulder=None, hip=None, ankle=None, hip_sag: float = 0.0):
        """Call on each frame with computed hip_angle (degrees) and optional points.

        Returns current state dict for display.
        """
        try:
            self.hip_angle_history.append(hip_angle)
            if len(self.hip_angle_history) > 100:
                self.hip_angle_history.pop(0)

            # update sag tracking
            self.current_min_sag = min(self.current_min_sag, hip_sag)
            self.current_max_sag = max(self.current_max_sag, hip_sag)
            self.current_hip_angles.append(hip_angle)

            # detect start with debounce (require consecutive frames)
            if hip_angle >= self.HOLD_ANGLE and self.stage == 'not_holding':
                self._hold_candidate_frames += 1
            else:
                self._hold_candidate_frames = 0

            if self._hold_candidate_frames >= self.HOLD_FRAMES and self.stage == 'not_holding':
                self.stage = 'holding'
                self.current_start = time.time()
                self.current_min_sag = hip_sag
                self.current_max_sag = hip_sag
                self.current_hip_angles = [hip_angle]
                self._hold_candidate_frames = 0

            # detect release with debounce
            if hip_angle <= self.RELEASE_ANGLE and self.stage == 'holding':
                self._release_candidate_frames += 1
            else:
                self._release_candidate_frames = 0

            if self._release_candidate_frames >= self.RELEASE_FRAMES and self.stage == 'holding':
                # finish hold
                end_t = time.time()
                duration = end_t - (self.current_start or end_t)
                alignment = self._calc_alignment_score(shoulder, hip, ankle) if shoulder is not None and hip is not None and ankle is not None else 100.0
                avg_hip_angle = float(np.mean(self.current_hip_angles)) if self.current_hip_angles else float(hip_angle)
                form_quality = self._calc_form_quality(avg_hip_angle, alignment, self.current_max_sag)

                self.counter += 1
                hold = PlankHold(
                    hold_number=self.counter,
                    start_time=self.current_start,
                    end_time=end_t,
                    duration=duration,
                    hip_angle=avg_hip_angle,
                    alignment=alignment,
                    hip_sag=self.current_max_sag,
                    form_quality=form_quality,
                )
                self.holds.append(hold)

                # reset
                self.stage = 'not_holding'
                self.current_start = None
                self.current_min_sag = float('inf')
                self.current_max_sag = 0
                self.current_hip_angles = []
                self._release_candidate_frames = 0

            # produce display state
            alignment = self._calc_alignment_score(shoulder, hip, ankle) if shoulder is not None and hip is not None and ankle is not None else 100.0
            current_form = self._calc_form_quality(hip_angle, alignment, hip_sag)

            return {
                'stage': self.stage,
                'counter': self.counter,
                'hip_angle': float(hip_angle),
                'alignment': float(alignment),
                'hip_sag': float(hip_sag),
                'form_quality': float(current_form),
            }
        except Exception as e:
            print('PlankAnalyzer update error:', e)
            return {
                'stage': self.stage,
                'counter': self.counter,
                'hip_angle': 0.0,
                'alignment': 100.0,
                'hip_sag': 0.0,
                'form_quality': 0.0,
            }

    def get_statistics(self):
        if not self.holds:
            return {}
        durations = [h.duration for h in self.holds]
        avg_hold = float(np.mean(durations))
        best = max(self.holds, key=lambda x: x.duration)
        worst = min(self.holds, key=lambda x: x.form_quality)
        avg_quality = float(np.mean([h.form_quality for h in self.holds]))
        avg_sag = float(np.mean([h.hip_sag for h in self.holds]))

        return {
            'total_holds': self.counter,
            'avg_hold_time': avg_hold,
            'best_hold': best,
            'worst_hold': worst,
            'avg_form_quality': avg_quality,
            'avg_hip_sag': avg_sag,
        }


def print_session_metrics(analyzer: PlankAnalyzer):
    stats = analyzer.get_statistics()
    if not stats:
        print('No holds detected in this session.')
        return
    best = stats.get('best_hold')
    worst = stats.get('worst_hold')
    print('\n=== PLANK SESSION METRICS ===')
    print(f"Total holds: {stats.get('total_holds', 0)}")
    print(f"Avg hold time: {stats.get('avg_hold_time', 0):.2f}s")
    print(f"Avg form quality: {stats.get('avg_form_quality', 0):.1f}/100")
    print(f"Avg hip sag: {stats.get('avg_hip_sag', 0):.1f}px")
    if best:
        print(f"Best hold: #{best.hold_number} - {best.duration:.2f}s | Quality: {best.form_quality:.1f}")
    if worst:
        print(f"Worst hold: #{worst.hold_number} - Quality: {worst.form_quality:.1f}")
    print('===============================\n')


def print_athlete_performance(analyzer: PlankAnalyzer):
    stats = analyzer.get_statistics()
    if not stats:
        print('No holds -> athlete performance unavailable.')
        return
    # More detailed performance analysis
    avg_quality = stats.get('avg_form_quality', 0)
    avg_hold = stats.get('avg_hold_time', 0)
    total_holds = stats.get('total_holds', 0)
    holds = analyzer.holds

    durations = [h.duration for h in holds] if holds else []
    qualities = [h.form_quality for h in holds] if holds else []

    import statistics
    consistency = 100.0
    if len(durations) > 1 and statistics.mean(durations) > 0:
        dur_std = float(np.std(durations))
        consistency = max(0.0, 100.0 - (dur_std / float(np.mean(durations))) * 100.0)

    percent_good = 0.0
    if holds:
        percent_good = sum(1 for h in holds if h.form_quality >= 75) / len(holds) * 100.0

    total_time = float(sum(durations))
    longest = max(durations) if durations else 0.0
    median = float(statistics.median(durations)) if durations else 0.0
    top3 = sorted(durations, reverse=True)[:3]

    # Compose weighted overall score
    # Weights: avg_quality 40%, avg_hold_time 30%, consistency 15%, percent_good 15%
    target_hold = 60.0
    hold_score = max(0.0, min(100.0, (avg_hold / target_hold) * 100.0))
    overall = (avg_quality * 0.40 + hold_score * 0.30 + consistency * 0.15 + percent_good * 0.15)
    overall = float(max(0.0, min(100.0, overall)))
    tier = 'Excellent' if overall >= 85 else 'Good' if overall >= 70 else 'Fair' if overall >= 50 else 'Needs Improvement'

    print('\n=== ATHLETE PERFORMANCE (PLANK) ===')
    print(f"Overall score: {overall:.1f}/100 ({tier})")
    print(f"Components -> Form: {avg_quality:.1f} | Avg hold: {avg_hold:.1f}s | Consistency: {consistency:.1f} | %Good holds: {percent_good:.1f}")
    print(f"Total holds: {total_holds} | Total hold time: {total_time:.1f}s | Longest: {longest:.1f}s | Median: {median:.1f}s")
    if top3:
        print(f"Top holds (s): {', '.join(f'{t:.1f}' for t in top3)}")

    # Recommendations
    print('\nRecommendations:')
    if avg_quality < 80:
        print(' - Improve form: work on maintaining a straight line from shoulders to ankles.')
    if avg_hold < 30:
        print(' - Increase endurance: practice sets of 30-60s holds.')
    if consistency < 70:
        print(' - Improve consistency: keep tempo and breathing steady between holds.')
    if percent_good < 60:
        print(' - Focus individual holds: aim for quality >75 on most holds.')

    print('===================================\n')


def export_performance_report(analyzer: PlankAnalyzer, path: str):
    """Export a JSON report with session statistics and per-hold details."""
    import json
    stats = analyzer.get_statistics()
    if not stats:
        print('No holds to export.')
        return

    report = {
        'summary': {
            'total_holds': stats.get('total_holds', 0),
            'avg_hold_time': stats.get('avg_hold_time', 0),
            'avg_form_quality': stats.get('avg_form_quality', 0),
            'avg_hip_sag': stats.get('avg_hip_sag', 0),
        },
        'holds': [
            {
                'hold_number': h.hold_number,
                'start_time': h.start_time,
                'end_time': h.end_time,
                'duration': h.duration,
                'form_quality': h.form_quality,
                'hip_sag': h.hip_sag,
            } for h in analyzer.holds
        ]
    }

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f'Report exported to: {path}')
    except Exception as e:
        print(f'Failed to export report: {e}')
