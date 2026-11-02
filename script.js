(() => {
  const video = document.getElementById('video');
  const stage = document.getElementById('stage');
  const drawCanvas = document.getElementById('drawCanvas');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const ctx = drawCanvas.getContext('2d');
  const octx = overlayCanvas.getContext('2d');
  const startBtn = document.getElementById('startBtn');
  const switchBtn = document.getElementById('switchBtn');
  const placeholder = document.getElementById('placeholder');
  const cameraLabel = document.getElementById('cameraLabel');
  const trackingStatus = document.getElementById('trackingStatus');
  const gestureStatus = document.getElementById('gestureStatus');
  const message = document.getElementById('message');
  const colorPicker = document.getElementById('colorPicker');
  const sizePicker = document.getElementById('sizePicker');
  const undoBtn = document.getElementById('undoBtn');
  const clearBtn = document.getElementById('clearBtn');

  const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  let landmarker = null;
  let visionApi = null;
  let stream = null;
  let camera = 'user';
  let mirrored = true;
  let running = false;
  let rafId = 0;
  let lastVideoTime = -1;
  let pinching = false;
  let lastPoint = null;
  let currentStroke = null;
  let strokes = [];
  let inferenceBusy = false;

  function resize() {
    const rect = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const old = document.createElement('canvas');
    old.width = drawCanvas.width;
    old.height = drawCanvas.height;
    if (old.width && old.height) old.getContext('2d').drawImage(drawCanvas, 0, 0);
    for (const canvas of [drawCanvas, overlayCanvas]) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (old.width) ctx.drawImage(old, 0, 0, old.width, old.height, 0, 0, rect.width, rect.height);
  }

  function redraw() {
    const r = stage.getBoundingClientRect();
    ctx.clearRect(0, 0, r.width, r.height);
    for (const stroke of strokes) {
      if (!stroke.points.length) continue;
      ctx.beginPath();
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = stroke.size;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      for (let i = 1; i < stroke.points.length; i++) ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
      if (stroke.points.length === 1) ctx.lineTo(stroke.points[0].x + 0.01, stroke.points[0].y + 0.01);
      ctx.stroke();
    }
    undoBtn.disabled = strokes.length === 0;
  }

  // video의 object-fit:cover 크롭을 반영해 랜드마크와 손끝 점을 정확히 겹칩니다.
  function toStagePoint(landmark) {
    const rect = stage.getBoundingClientRect();
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const scale = Math.max(rect.width / vw, rect.height / vh);
    const shownW = vw * scale;
    const shownH = vh * scale;
    const offsetX = (rect.width - shownW) / 2;
    const offsetY = (rect.height - shownH) / 2;
    return {
      x: landmark.x * shownW + offsetX,
      y: landmark.y * shownH + offsetY
    };
  }

  async function loadLandmarker() {
    if (landmarker) return landmarker;
    message.textContent = '손 인식 AI 모델을 불러오는 중…';
    // 동적 import로 라이브러리 로딩 실패도 앱 내부 오류 메시지로 표시합니다.
    if (!visionApi) {
      visionApi = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32');
    }
    const { FilesetResolver, HandLandmarker } = visionApi;
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const options = {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
    };
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, options);
    } catch (gpuError) {
      console.warn('GPU 초기화 실패, CPU로 다시 시도합니다.', gpuError);
      options.baseOptions.delegate = 'CPU';
      landmarker = await HandLandmarker.createFromOptions(vision, options);
    }
    return landmarker;
  }

  function stopCamera() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    pinching = false;
    lastPoint = null;
    currentStroke = null;
  }

  async function startCamera() {
    startBtn.disabled = true;
    message.textContent = '카메라와 손 인식 모델을 준비하는 중…';
    try {
      stopCamera();
      await loadLandmarker();
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: camera },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      video.srcObject = stream;
      await video.play();
      if (!video.videoWidth) {
        await new Promise((resolve, reject) => {
          video.onloadedmetadata = resolve;
          video.onerror = reject;
        });
      }
      const track = stream.getVideoTracks()[0];
      const actualFacing = track.getSettings().facingMode;
      mirrored = actualFacing ? actualFacing !== 'environment' : camera === 'user';
      video.style.transform = mirrored ? 'scaleX(-1)' : 'none';
      drawCanvas.style.transform = mirrored ? 'scaleX(-1)' : 'none';
      overlayCanvas.style.transform = mirrored ? 'scaleX(-1)' : 'none';
      resize();
      placeholder.classList.add('hidden');
      switchBtn.disabled = false;
      cameraLabel.textContent = mirrored ? '전면 카메라' : '후면 카메라';
      message.textContent = '손을 화면 안에 비춰보세요. 검지와 엄지 끝에 표시점이 붙어 움직입니다.';
      running = true;
      lastVideoTime = -1;
      processFrame();
    } catch (err) {
      console.error(err);
      stopCamera();
      placeholder.classList.remove('hidden');
      cameraLabel.textContent = '시작 오류';
      const detail = err?.message ? ' (' + err.message + ')' : '';
      message.textContent = err?.name === 'NotAllowedError'
        ? '카메라 권한을 허용해 주세요.'
        : '카메라 또는 손 인식 모델을 시작하지 못했습니다. 인터넷 연결과 권한을 확인해 주세요.' + detail;
    } finally {
      startBtn.disabled = false;
    }
  }

  function processFrame() {
    if (!running) return;
    if (!inferenceBusy && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      inferenceBusy = true;
      lastVideoTime = video.currentTime;
      try {
        const result = landmarker.detectForVideo(video, performance.now());
        onResults(result);
      } catch (err) {
        console.error('손 인식 중 오류:', err);
        trackingStatus.textContent = '● 인식 오류';
        gestureStatus.textContent = '손 인식 처리 중 오류가 발생했습니다';
        message.textContent = '손 인식 오류: ' + (err.message || '알 수 없는 오류');
      } finally {
        inferenceBusy = false;
      }
    }
    rafId = requestAnimationFrame(processFrame);
  }

  function onResults(result) {
    const rect = stage.getBoundingClientRect();
    octx.clearRect(0, 0, rect.width, rect.height);
    const hand = result.landmarks?.[0];
    if (!hand || hand.length < 21) {
      pinching = false;
      lastPoint = null;
      currentStroke = null;
      trackingStatus.textContent = '● 손 인식 대기';
      trackingStatus.classList.remove('active');
      gestureStatus.textContent = '손을 카메라 안에 보여주세요';
      return;
    }

    // MediaPipe 21개 관절 중 4번은 엄지 끝, 8번은 검지 끝입니다.
    const thumb = toStagePoint(hand[4]);
    const index = toStagePoint(hand[8]);
    const wrist = toStagePoint(hand[0]);
    const middleBase = toStagePoint(hand[9]);
    const gap = Math.hypot(thumb.x - index.x, thumb.y - index.y);
    const palm = Math.max(1, Math.hypot(wrist.x - middleBase.x, wrist.y - middleBase.y));
    const ratio = gap / palm;
    const shouldPinch = pinching ? ratio < 0.48 : ratio < 0.34;

    // 엄지 끝은 주황색, 검지 끝은 파란색. 매 프레임 위치를 새로 계산해 손끝을 따라갑니다.
    function fingertipDot(point, color, label) {
      octx.beginPath();
      octx.arc(point.x, point.y, 8, 0, Math.PI * 2);
      octx.fillStyle = color;
      octx.fill();
      octx.lineWidth = 2.5;
      octx.strokeStyle = '#ffffff';
      octx.stroke();
      octx.font = '600 11px system-ui, sans-serif';
      octx.textAlign = 'center';
      octx.fillStyle = '#ffffff';
      octx.shadowColor = '#000000';
      octx.shadowBlur = 4;
      octx.fillText(label, point.x, point.y - 15);
      octx.shadowBlur = 0;
    }
    fingertipDot(thumb, '#ff9f43', '엄지');
    fingertipDot(index, '#6695ff', '검지');

    octx.beginPath();
    octx.moveTo(thumb.x, thumb.y);
    octx.lineTo(index.x, index.y);
    octx.strokeStyle = shouldPinch ? '#83e6b0' : '#ffffff80';
    octx.lineWidth = 2;
    octx.stroke();

    if (shouldPinch) {
      if (!pinching) {
        currentStroke = { color: colorPicker.value, size: Number(sizePicker.value), points: [] };
        strokes.push(currentStroke);
      }
      const drawPoint = index;
      if (lastPoint) {
        const smooth = {
          x: lastPoint.x * 0.35 + drawPoint.x * 0.65,
          y: lastPoint.y * 0.35 + drawPoint.y * 0.65
        };
        currentStroke.points.push(smooth);
        lastPoint = smooth;
      } else {
        currentStroke.points.push(drawPoint);
        lastPoint = drawPoint;
      }
      redraw();
      gestureStatus.textContent = '그리는 중 · 엄지와 검지 접촉';
    } else {
      lastPoint = null;
      currentStroke = null;
      gestureStatus.textContent = '엄지와 검지를 모으면 그려집니다';
    }
    pinching = shouldPinch;
    trackingStatus.textContent = '● 손 추적 중';
    trackingStatus.classList.add('active');
  }

  startBtn.addEventListener('click', startCamera);
  switchBtn.addEventListener('click', async () => {
    camera = camera === 'user' ? 'environment' : 'user';
    await startCamera();
  });
  undoBtn.addEventListener('click', () => {
    strokes.pop();
    redraw();
  });
  clearBtn.addEventListener('click', () => {
    strokes = [];
    currentStroke = null;
    lastPoint = null;
    pinching = false;
    redraw();
  });
  window.addEventListener('resize', () => {
    resize();
    redraw();
  });
  window.addEventListener('beforeunload', stopCamera);

  if (!navigator.mediaDevices?.getUserMedia) {
    message.textContent = '이 브라우저는 카메라 기능을 지원하지 않습니다. HTTPS 주소에서 최신 Chrome으로 열어주세요.';
    startBtn.disabled = true;
  }
  if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
    window.addEventListener('load', () => navigator.serviceWorker.register('./sw.js').catch(console.warn));
  }
})();