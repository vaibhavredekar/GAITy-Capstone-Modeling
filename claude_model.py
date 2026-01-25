#!/usr/bin/env python3
"""
GAIT ANALYSIS STUDIO - PRODUCTION GRADE
Complete Pipeline: Video → Features → AI → Clinical Analysis
Single-File Architecture | Version 3.0
"""

import os, sys, warnings, logging, traceback, hashlib, json, subprocess, zipfile
from io import BytesIO
from typing import Optional, Dict, Tuple, List
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import importlib.util

os.environ.update({'TF_ENABLE_ONEDNN_OPTS':'0','TF_CPP_MIN_LOG_LEVEL':'3','PYTHONWARNINGS':'ignore'})
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, resample
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
    handlers=[logging.FileHandler("gait_studio.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"▶ START: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✓ END: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"✗ ERROR: {func.__name__} | {e}\n{traceback.format_exc()}")
            raise
    return wrapper

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
MEDIAPIPE_SCRIPT = ROOT / "pre-processing-models" / "mediapipe" / "pre_mediapipe.py"

UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "output"
FEATURES_DIR = ROOT / "data" / "features"
GAIT_CYCLES_DIR = ROOT / "data" / "gait_cycles"
MODELS_DIR = ROOT / "models" / "baseline"

BINARY_MODEL = MODELS_DIR / "xgboost_model.bin"
BINARY_FEATURES = MODELS_DIR / "feature_names.json"
MULTICLASS_MODEL = MODELS_DIR / "xgboost_gait_5class.bin"
MULTICLASS_META = MODELS_DIR / "xgboost_gait_5class_metadata.json"

for d in [UPLOAD_DIR, OUTPUT_DIR, FEATURES_DIR, GAIT_CYCLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    predictions: List[str]
    probabilities: List[float]
    details: Optional[pd.DataFrame] = None

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO CONVERTER
# ═══════════════════════════════════════════════════════════════════════════

class VideoConverter:
    @staticmethod
    def check_ffmpeg():
        try: return subprocess.run(['ffmpeg','-version'], capture_output=True, timeout=5).returncode == 0
        except: return False
    
    @staticmethod
    def get_codec(path):
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            cap.release()
            return "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)]).strip()
        except: return None
    
    @staticmethod
    def is_web_compatible(codec):
        return bool(codec and any(c in codec.upper() for c in ['AVC1','H264','X264']))
    
    @staticmethod
    @log_execution
    def convert(inp, out):
        try:
            cmd = ['ffmpeg','-i',str(inp),'-c:v','libx264','-preset','medium','-crf','23',
                   '-pix_fmt','yuv420p','-movflags','+faststart','-c:a','aac','-b:a','128k','-y',str(out)]
            return subprocess.run(cmd, capture_output=True, timeout=300).returncode == 0 and out.exists()
        except: return False
    
    @staticmethod
    def ensure_web(path):
        if not path or not path.exists(): return path
        web = path.parent / f"{path.stem}_h264.mp4"
        if web.exists(): return web
        if VideoConverter.is_web_compatible(VideoConverter.get_codec(path)): return path
        if VideoConverter.check_ffmpeg() and VideoConverter.convert(path, web): return web
        return path

# ═══════════════════════════════════════════════════════════════════════════
# FILE MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class FileManager:
    @staticmethod
    @log_execution
    def save_video(uploaded):
        try:
            data = uploaded.getvalue()
            h = hashlib.md5(data).hexdigest()
            name, ext = Path(uploaded.name).stem, Path(uploaded.name).suffix
            path = UPLOAD_DIR / f"{name}{ext}"
            
            if path.exists():
                with open(path,'rb') as f:
                    if hashlib.md5(f.read()).hexdigest() == h: return path, True
            
            with open(path,'wb') as f: f.write(data)
            
            import cv2
            cap = cv2.VideoCapture(str(path))
            ok = cap.isOpened()
            cap.release()
            if not ok: path.unlink(); return None, False
            
            return path, False
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return None, False
    
    @staticmethod
    @log_execution
    def find_outputs(video_path):
        results = {'annotated':None,'skeleton':None,'csv':None}
        stem = video_path.stem.lower()
        files = list(OUTPUT_DIR.glob("*"))
        
        def match(patterns):
            for f in files:
                if stem in f.stem.lower():
                    if any(p in f.stem.lower() for p in patterns): return f
            return None
        
        results['csv'] = match(['landmarks','landmark'])
        results['annotated'] = match(['annotated'])
        results['skeleton'] = match(['skeleton'])
        
        if not results['csv']:
            csvs = [f for f in files if 'landmark' in f.stem.lower() and f.suffix=='.csv']
            if csvs: results['csv'] = sorted(csvs, key=lambda x:x.stat().st_mtime, reverse=True)[0]
        
        logger.info(f"Found: {[(k,v.name if v else None) for k,v in results.items()]}")
        return results

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class Pipeline:
    @staticmethod
    def load_config():
        if not CONFIG_PATH.exists(): return {}
        try:
            with open(CONFIG_PATH) as f: return json.load(f)
        except: return None
    
    @staticmethod
    def save_config(cfg):
        try:
            with open(CONFIG_PATH,'w') as f: json.dump(cfg,f,indent=2)
            return True
        except: return False
    
    @staticmethod
    def update_config(path):
        cfg = Pipeline.load_config() or {}
        cfg["input_paths"] = [str(path)]
        cfg["output_dir"] = "data/output"
        return Pipeline.save_config(cfg)
    
    @staticmethod
    @log_execution
    def run():
        if not MEDIAPIPE_SCRIPT.exists():
            st.error("MediaPipe script not found")
            return None
        try:
            spec = importlib.util.spec_from_file_location("mp", MEDIAPIPE_SCRIPT)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.main() if hasattr(mod,'main') else {"status":"loaded"}
        except ImportError as e:
            if "DLL" in str(e):
                st.error("⚠️ TensorFlow/MediaPipe DLL error. Reinstall dependencies.")
            else: st.error(f"Import error: {e}")
            return None
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════════════
# GAIT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class Gait:
    N,LH,RH,LS,RS,LK,RK,LA,RA = 33,23,24,11,12,25,26,27,28
    
    @staticmethod
    @log_execution
    def interp(pose):
        T,J,C = pose.shape
        out = pose.copy()
        for j in range(J):
            for c in range(C):
                coord, missing = pose[:,j,c], pose[:,j,c]==0
                if not missing.all():
                    valid = np.where(~missing)[0]
                    out[:,j,c] = np.interp(np.arange(T), valid, coord[valid])
        return out
    
    @staticmethod
    @log_execution
    def normalize(pose):
        pelvis = (pose[:,Gait.LH] + pose[:,Gait.RH])/2
        cent = pose - pelvis[:,None,:]
        torso = (cent[:,Gait.LS] + cent[:,Gait.RS])/2
        scale = np.linalg.norm(torso,axis=1).mean()
        return cent / (scale if scale>0 and np.isfinite(scale) else 1.0)
    
    @staticmethod
    @log_execution
    def add_pose_col(df):
        rows = []
        for vid in df["video_id"].unique():
            g = df[df["video_id"]==vid].sort_values("frame")
            frames = np.sort(g["frame"].unique())
            fidx = {f:i for i,f in enumerate(frames)}
            T = len(frames)
            pose = np.zeros((T,Gait.N,3),dtype=np.float32)
            for _,r in g.iterrows():
                fi,j = fidx[r["frame"]], int(r["landmark_id"])
                if j<Gait.N: pose[fi,j,:] = [r["x_norm"],r["y_norm"],r["z_norm"]]
            pose = Gait.interp(pose)
            row = {c:g.iloc[0].get(c) for c in ['video_id','fps'] if c in g.columns}
            row["pose"] = pose
            rows.append(row)
        return pd.DataFrame(rows)
    
    @staticmethod
    def windows(pose, fps=30, sec=2.0, overlap=0.5):
        T,_,_ = pose.shape
        wf, sf = int(sec*fps), int(sec*fps*(1-overlap))
        if wf>T: return [pose]
        return [pose[s:s+wf] for s in range(0,T-wf+1,sf)]
    
    @staticmethod
    @log_execution
    def preprocess(df, sec=2.0, overlap=0.5, frames=60):
        wins, ids = [], []
        for _,r in df.iterrows():
            pose = np.asarray(r.get("pose"))
            if pose is None or pose.ndim!=3: continue
            fps, vid = r.get("fps",30), r.get("video_id","v")
            try: pn = Gait.normalize(pose)
            except: continue
            for i,w in enumerate(Gait.windows(pn,fps,sec,overlap)):
                if w.shape[0]<2: continue
                wr = resample(w,frames,axis=0)
                wins.append(wr)
                ids.append(f"{vid}_w{i}")
        return np.array(wins), ids
    
    @staticmethod
    def qc(win):
        pelvis = (win[:,Gait.LH]+win[:,Gait.RH])/2
        off = np.linalg.norm(pelvis.mean(axis=0))
        torso_std = np.linalg.norm(win[:,Gait.LS]-pelvis,axis=1).std()
        peaks,_ = find_peaks(win[:,Gait.LA,1], distance=24)
        fail = off>0.1 or torso_std>0.15 or len(peaks)<1
        return {"qc_fail":fail}
    
    @staticmethod
    @log_execution
    def apply_qc(X, ids):
        clean, cids = [], []
        for i,w in enumerate(X):
            if not Gait.qc(w)["qc_fail"]:
                clean.append(w)
                cids.append(ids[i])
        return np.array(clean), cids
    
    @staticmethod
    def features(win, fps=60):
        pelvis = (win[:,Gait.LH]+win[:,Gait.RH])/2
        if np.linalg.norm(pelvis.mean(axis=0))>1e-2: win = Gait.normalize(win)
        
        f = {}
        rom = lambda j,ax: float(win[:,j,ax].max()-win[:,j,ax].min())
        f["step_height_L"] = rom(Gait.LA,1)
        f["step_height_R"] = rom(Gait.RA,1)
        f["step_length_L"] = rom(Gait.LA,0)
        f["step_length_R"] = rom(Gait.RA,0)
        eps = 1e-6
        hL,hR = f["step_height_L"], f["step_height_R"]
        f["step_height_symmetry"] = (hL-hR)/(hL+hR+eps)
        
        traj = gaussian_filter1d(win[:,Gait.LA,:], sigma=1.0, axis=0)
        speed = np.linalg.norm(np.diff(traj,axis=0),axis=1)*fps
        f["ankle_L_moving_fraction"] = float((speed>0.02).mean())
        
        def angle(p1,p2,p3):
            v1,v2 = p1-p2, p3-p2
            cos = np.sum(v1*v2,axis=1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)+1e-6)
            return np.degrees(np.arccos(np.clip(cos,-1,1)))
        
        ka = angle(win[:,Gait.LH], win[:,Gait.LK], win[:,Gait.LA])
        f["knee_angle_L_mean"] = float(ka.mean())
        f["knee_angle_L_rom"] = float(ka.max()-ka.min())
        return f
    
    @staticmethod
    @log_execution
    def extract_all(X):
        return pd.DataFrame([Gait.features(X[i]) for i in range(len(X))])
    
    @staticmethod
    @log_execution
    def from_csv(path):
        df = pd.read_csv(path)
        if 'video_id' not in df.columns: df['video_id']='default'
        dfv = Gait.add_pose_col(df)
        X, ids = Gait.preprocess(dfv)
        Xc, idsc = Gait.apply_qc(X, ids)
        if len(Xc)==0: return pd.DataFrame(), None
        feat = Gait.extract_all(Xc)
        feat['window_id'] = idsc
        return feat, Xc

# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════

class Model:
    def __init__(self):
        self.bin_model = self.multi_model = None
        self.bin_feats = self.multi_meta = self.multi_classes = None
    
    @log_execution
    def load_binary(self):
        if not BINARY_MODEL.exists() or not BINARY_FEATURES.exists(): return False
        try:
            self.bin_model = xgb.XGBClassifier()
            self.bin_model.load_model(BINARY_MODEL)
            with open(BINARY_FEATURES) as f: self.bin_feats = json.load(f)
            return True
        except: return False
    
    @log_execution
    def load_multi(self):
        if not MULTICLASS_MODEL.exists() or not MULTICLASS_META.exists(): return False
        try:
            self.multi_model = xgb.XGBClassifier()
            self.multi_model.load_model(MULTICLASS_MODEL)
            with open(MULTICLASS_META) as f: self.multi_meta = json.load(f)
            id2cls = self.multi_meta.get("id_to_class",{})
            self.multi_classes = [id2cls[str(i)] for i in sorted(map(int,id2cls.keys()))]
            return True
        except: return False
    
    def align(self, df, req):
        for m in set(req)-set(df.columns): df[m]=0.0
        return df[req]
    
    @log_execution
    def predict_bin(self, df):
        if not self.bin_model: return None
        aln = self.align(df.copy(), self.bin_feats).fillna(0)
        probs = self.bin_model.predict_proba(aln)
        preds = self.bin_model.predict(aln)
        labels = ["Normal" if p==0 else "Abnormal" for p in preds]
        details = df.copy()
        details['prediction'] = labels
        details['confidence'] = [max(p) for p in probs]
        return PredictionResult(labels, [max(p) for p in probs], details)
    
    @log_execution
    def predict_multi(self, df):
        if not self.multi_model: return None
        req = self.multi_meta.get("feature_cols",[])
        aln = self.align(df.copy(), req).fillna(0)
        probs = self.multi_model.predict_proba(aln)
        preds = self.multi_model.predict(aln)
        labels = [self.multi_classes[p] for p in preds]
        details = df.copy()
        details['prediction'] = labels
        details['confidence'] = [probs[i,p] for i,p in enumerate(preds)]
        return PredictionResult(labels, [probs[i,p] for i,p in enumerate(preds)], details)

# ═══════════════════════════════════════════════════════════════════════════
# VIZ
# ═══════════════════════════════════════════════════════════════════════════

class Viz:
    @staticmethod
    def dashboard(df):
        fig,ax = plt.subplots(figsize=(12,8))
        metrics = {'Step Symmetry':('step_height_symmetry',0.15,0.25),
                   'Knee Flexion':('knee_angle_L_rom',30,60),
                   'Ankle Control':('ankle_L_moving_fraction',0.3,0.7)}
        pos = [(0.2,0.8),(0.5,0.8),(0.8,0.8)]
        for i,(name,(key,mn,mx)) in enumerate(metrics.items()):
            x,y = pos[i]
            if key in df.columns:
                val = df.iloc[0][key]
                col = 'green' if mn<=val<=mx else 'red'
                ax.add_patch(Circle((x,y),0.08,color=col,alpha=0.8))
                ax.text(x,y-0.15,name,ha='center',fontsize=10,weight='bold')
                ax.text(x,y,f'{val:.2f}',ha='center',va='center',fontsize=9,color='white',weight='bold')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title('Gait Health Dashboard',fontsize=16,weight='bold')
        ax.axis('off')
        return fig
    
    @staticmethod
    def trajectory_3d(cycles):
        if cycles is None or len(cycles)==0: return None
        fig = plt.figure(figsize=(15,10))
        avg = np.mean(cycles,axis=0)
        joints = [27,28,25,26,23,24]
        names = ['L Ankle','R Ankle','L Knee','R Knee','L Hip','R Hip']
        views = [(0,0),(0,90),(90,0),(30,45)]
        labels = ['Front','Side','Top','3D']
        for idx,((e,a),lbl) in enumerate(zip(views,labels)):
            ax = fig.add_subplot(2,2,idx+1,projection='3d')
            for j,n in zip(joints,names):
                t = avg[:,j,:]
                c = 'blue' if 'L' in n else 'red'
                ax.plot(t[:,0],t[:,1],t[:,2],color=c,linewidth=2,label=n)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title(lbl); ax.view_init(elev=e,azim=a)
        plt.suptitle('3D Joint Trajectories',fontsize=16)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def pred_chart(res, typ="binary"):
        if not res: return None
        fig,ax = plt.subplots(1,2,figsize=(12,5))
        labels, counts = np.unique(res.predictions, return_counts=True)
        if typ=="binary":
            colors = ['green' if 'Normal' in l else 'red' for l in labels]
            ax[0].bar(labels,counts,color=colors)
            ax[0].set_title("Predictions")
            ax[1].hist(res.probabilities,bins=10,color='skyblue',edgecolor='black')
            ax[1].set_title("Confidence")
        else:
            ax[0].barh(labels,counts,color='teal')
            ax[0].set_title("Predictions")
        plt.tight_layout()
        return fig

# ═══════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════

def init():
    if 'predictor' not in st.session_state: st.session_state.predictor = Model()
    defaults = {'video':None,'done':False,'outs':{},'feat':None,'cycles':None,'csv_feat':None,'pb':None,'pm':None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    st.set_page_config(page_title="Gait Studio", layout="wide")
    st.markdown("""<style>
    .stApp {background-color: #F5F5F7;}
    h1,h2,h3 {color: #1F2937; font-family: 'Segoe UI'; font-weight: 600;}
    .stButton>button {background-color: #2C3E50; color: white; border-radius: 5px; font-weight: bold;}
    </style>""", unsafe_allow_html=True)
    
    init()
    
    st.title("🚶 Gait Analysis Studio")
    st.markdown("### Complete Pipeline: Processing → Features → AI → Analysis")
    
    with st.sidebar:
        st.header("⚙️ Control")
        if BINARY_MODEL.exists(): st.success("✅ Binary Model")
        else: st.warning("⚠️ Binary Missing")
        if MULTICLASS_MODEL.exists(): st.success("✅ Multi-class Model")
        else: st.warning("⚠️ Multi Missing")
        st.markdown("---")
        if st.button("🔄 Reset"): 
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        with st.expander("📋 Logs"):
            try:
                with open("gait_studio.log") as f:
                    st.text_area("", f.read()[-2000:], height=300)
            except: st.text("No logs yet")
    
    t1,t2,t3,t4,t5,t6 = st.tabs(["📤 Upload","⚙️ Process","🎬 Videos","📊 Features","🔬 Analysis","🤖 AI"])
    
    with t1:
        up = st.file_uploader("Upload Video", type=["mp4","mov","avi"])
        if up:
            path, dup = FileManager.save_video(up)
            if path:
                st.session_state.video = path
                st.success(f"✅ Saved: {path.name}")
                if dup: st.info("⚠️ Duplicate")
                web = VideoConverter.ensure_web(path)
                with open(web,'rb') as v: st.video(v.read())
    
    with t2:
        if not st.session_state.video: st.warning("Upload video first")
        else:
            st.info(f"Ready: {st.session_state.video.name}")
            col1,col2 = st.columns(2)
            with col1:
                if st.button("📝 Update Config"):
                    if Pipeline.update_config(st.session_state.video): st.success("Updated")
                    else: st.error("Failed")
            with col2:
                if st.button("▶️ Run Pipeline"):
                    with st.spinner("Processing..."):
                        Pipeline.update_config(st.session_state.video)
                        if Pipeline.run():
                            st.session_state.done = True
                            st.session_state.outs = FileManager.find_outputs(st.session_state.video)
                            st.success("Done!"); st.balloons()
                        else: st.error("Failed")
    
    with t3:
        if not st.session_state.done: st.warning("Process video first")
        else:
            if st.button("🔄 Refresh"): 
                st.session_state.outs = FileManager.find_outputs(st.session_state.video)
                st.rerun()
            v = st.session_state.outs
            with st.expander("🛠 Debug"): st.json(v)
            c1,c2 = st.columns(2)
            with c1:
                st.subheader("Annotated")
                if v.get('annotated'):
                    web = VideoConverter.ensure_web(v['annotated'])
                    with open(web,'rb') as f: st.video(f.read())
                else: st.warning("Not found")
            with c2:
                st.subheader("Skeleton")
                if v.get('skeleton'):
                    web = VideoConverter.ensure_web(v['skeleton'])
                    with open(web,'rb') as f: st.video(f.read())
                else: st.warning("Not found")
            if v.get('csv'):
                st.subheader("Landmarks")
                st.dataframe(pd.read_csv(v['csv']).head())
            else: st.error("CSV not found")
    
    with t4:
        sub1,sub2 = st.tabs(["From Video","From CSV"])
        with sub1:
            if st.session_state.done and st.session_state.outs.get('csv'):
                if st.button("🚀 Extract Features"):
                    with st.spinner("Extracting..."):
                        try:
                            feat,cyc = Gait.from_csv(st.session_state.outs['csv'])
                            if not feat.empty:
                                st.session_state.feat = feat
                                st.session_state.cycles = cyc
                                st.success(f"✅ {len(feat)} windows")
                            else: st.error("Failed")
                        except Exception as e: st.error(f"{e}")
            else: st.warning("Process video first")
            if st.session_state.feat is not None: st.dataframe(st.session_state.feat)
        
        with sub2:
            csv = st.file_uploader("Upload CSV", type=["csv"])
            if csv:
                try:
                    df = pd.read_csv(csv)
                    st.session_state.csv_feat = df
                    st.success(f"✅ {len(df)} rows")
                    st.dataframe(df.head())
                except Exception as e: st.error(f"{e}")
    
    with t5:
        target = st.session_state.feat if st.session_state.feat is not None else st.session_state.csv_feat
        cycles = st.session_state.cycles
        if target is None: st.warning("⚠️ Extract features first")
        else:
            st.success(f"✅ Analyzing {len(target)} vectors")
            viz = st.selectbox("Select:", ["Dashboard","3D Trajectories"])
            if viz=="Dashboard": st.pyplot(Viz.dashboard(target)) 
            elif viz=="3D Trajectories":
                        if cycles is not None and len(cycles) > 0:
                            st.pyplot(Viz.trajectory_3d(cycles))
                        else:
                            st.warning("No gait cycles available for 3D visualization")
    
    with t6:
        target = st.session_state.feat if st.session_state.feat is not None else st.session_state.csv_feat
        if target is None:
            st.warning("⚠️ Extract features first")
        else:
            st.success(f"✅ Ready to predict on {len(target)} samples")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Binary Classification")
                if st.button("Run Binary Prediction"):
                    if not st.session_state.predictor.load_binary():
                        st.error("Failed to load binary model")
                    else:
                        with st.spinner("Predicting..."):
                            result = st.session_state.predictor.predict_bin(target)
                            if result:
                                st.session_state.pb = result
                                st.success("✅ Prediction complete!")
                            else:
                                st.error("Prediction failed")
                
                if st.session_state.pb:
                    st.markdown("### Results")
                    unique, counts = np.unique(st.session_state.pb.predictions, return_counts=True)
                    for label, count in zip(unique, counts):
                        st.metric(label, count)
                    
                    st.pyplot(Viz.pred_chart(st.session_state.pb, "binary"))
                    
                    with st.expander("📊 Detailed Results"):
                        st.dataframe(st.session_state.pb.details)
                    
                    csv_buffer = BytesIO()
                    st.session_state.pb.details.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "💾 Download Results",
                        csv_buffer.getvalue(),
                        "binary_predictions.csv",
                        "text/csv"
                    )
            
            with col2:
                st.subheader("🎯 Multi-class Classification")
                if st.button("Run Multi-class Prediction"):
                    if not st.session_state.predictor.load_multi():
                        st.error("Failed to load multi-class model")
                    else:
                        with st.spinner("Predicting..."):
                            result = st.session_state.predictor.predict_multi(target)
                            if result:
                                st.session_state.pm = result
                                st.success("✅ Prediction complete!")
                            else:
                                st.error("Prediction failed")
                
                if st.session_state.pm:
                    st.markdown("### Results")
                    unique, counts = np.unique(st.session_state.pm.predictions, return_counts=True)
                    for label, count in zip(unique, counts):
                        st.metric(label, count)
                    
                    st.pyplot(Viz.pred_chart(st.session_state.pm, "multi"))
                    
                    with st.expander("📊 Detailed Results"):
                        st.dataframe(st.session_state.pm.details)
                    
                    csv_buffer = BytesIO()
                    st.session_state.pm.details.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "💾 Download Results",
                        csv_buffer.getvalue(),
                        "multiclass_predictions.csv",
                        "text/csv"
                    )
            
            # Combined Analysis Section
            if st.session_state.pb or st.session_state.pm:
                st.markdown("---")
                st.subheader("📈 Comparative Analysis")
                
                if st.session_state.pb and st.session_state.pm:
                    comparison_data = {
                        'Window ID': target.get('window_id', range(len(target))),
                        'Binary': st.session_state.pb.predictions,
                        'Binary Confidence': st.session_state.pb.probabilities,
                        'Multi-class': st.session_state.pm.predictions,
                        'Multi-class Confidence': st.session_state.pm.probabilities
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    st.dataframe(comparison_df)
                    
                    # Visualize agreement
                    fig, ax = plt.subplots(figsize=(10, 6))
                    binary_normal = [1 if p == "Normal" else 0 for p in st.session_state.pb.predictions]
                    ax.scatter(range(len(binary_normal)), binary_normal, 
                              c=st.session_state.pb.probabilities, 
                              cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Binary Classification (0=Abnormal, 1=Normal)')
                    ax.set_title('Binary Predictions with Confidence')
                    plt.colorbar(ax.collections[0], ax=ax, label='Confidence')
                    st.pyplot(fig)
                    
                    csv_buffer = BytesIO()
                    comparison_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "💾 Download Comparison",
                        csv_buffer.getvalue(),
                        "comparison_results.csv",
                        "text/csv"
                    )

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
