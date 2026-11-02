(() => {
  const video = document.getElementById('video');
  const stage = document.getElementById('stage');
  const drawCanvas = document.getElementById('drawCanvas');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const ctx = drawCanvas.getContext('2d');
  const octx = overlayCanvas.getContext('2d');
  const startBtn = document.getElementById('startBtn');
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
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
  ];

  let landmarker = null;
  let visionApi = null;
  let stream = null;
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

  // object-fit: cover로 잘린 영상 영역까지 계산해 관절 위치와 화면을 일치시킵니다.
  function toStagePoint(landmark) {
    const rect = stage.getBoundingClientRect();
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const scale = Math.max(rect.width / vw, rect.height / vh);
    const shownW = vw * scale;
    const shownH = vh * scale;
    return {
      x: landmark.x * shownW + (rect.width - shownW) / 2,
      y: landmark.y * shownH + (rect.height - shownH) / 2
    };
  }

  async function loadLandmarker() {
    if (landmarker) return landmarker;
    message.textContent = '손 관절 인식 AI 모델을 불러오는 중…';
    if (!visionApi) {
      visionApi = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs');
    }
    const { FilesetResolver, HandLandmarker } = visionApi;
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const options = {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35
    };
    landmarker = await HandLandmarker.createFromOptions(vision, options);
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
    message.textContent = '후면 카메라를 여는 중…';
    cameraLabel.textContent = '후면 카메라 연결 중';
    try {
      stopCamera();

      // AI 모델보다 카메라를 먼저 시작해 영상부터 표시합니다.
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('이 브라우저에서는 카메라 기능을 사용할 수 없습니다. HTTPS 주소의 최신 Chrome에서 열어주세요.');
      }

      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { exact: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });

      video.srcObject = stream;
      await video.play();

      if (!video.videoWidth) {
        await new Promise((resolve, reject) => {
          const onReady = () => {
            cleanup();
            resolve();
          };
          const onError = () => {
            cleanup();
            reject(new Error('카메라 영상 정보를 가져오지 못했습니다.'));
          };
          const cleanup = () => {
            video.removeEventListener('loadedmetadata', onReady);
            video.removeEventListener('error', onError);
          };
          video.addEventListener('loadedmetadata', onReady, { once: true });
          video.addEventListener('error', onError, { once: true });
          if (video.videoWidth) onReady();
        });
      }

      const track = stream.getVideoTracks()[0];
      const actualFacing = track.getSettings().facingMode;
      if (actualFacing && actualFacing !== 'environment') {
        throw new Error('후면 카메라를 확인하지 못했습니다. 기기의 후면 카메라 설정을 확인해 주세요.');
      }

      video.style.transform = 'none';
      drawCanvas.style.transform = 'none';
      overlayCanvas.style.transform = 'none';
      resize();
      placeholder.classList.add('hidden');
      cameraLabel.textContent = '후면 카메라 연결됨';
      message.textContent = '카메라 연결 성공 · 손 인식 AI 모델을 불러오는 중…';

      await loadLandmarker();

      message.textContent = '영상 ' + video.videoWidth + '×' + video.videoHeight + ' · 손 관절 추적 준비 완료';
      running = true;
      lastVideoTime = -1;
      processFrame();
    } catch (err) {
      console.error(err);
      stopCamera();
      placeholder.classList.remove('hidden');
      cameraLabel.textContent = '후면 카메라 시작 오류';
      const detail = err?.message ? ' (' + err.message + ')' : '';
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        message.textContent = '카메라 권한이 거부됐습니다. Chrome 사이트 설정에서 카메라 권한을 허용해 주세요.';
      } else if (err?.name === 'OverconstrainedError' || err?.name === 'NotFoundError') {
        message.textContent = '후면 카메라를 찾지 못했습니다. 기기의 카메라를 확인해 주세요.';
      } else if (err?.name === 'NotReadableError') {
        message.textContent = '카메라가 다른 앱에서 사용 중일 수 있습니다. 다른 카메라 앱을 닫고 다시 시도해 주세요.';
      } else {
        message.textContent = '카메라 또는 손 인식 AI를 시작하지 못했습니다.' + detail;
      }
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

  function drawHandSkeleton(points) {
    // 손목과 각 손가락의 21개 관절을 뼈대 선으로 연결합니다.
    octx.lineCap = 'round';
    octx.lineJoin = 'round';
    octx.lineWidth = 3;
    octx.strokeStyle = '#a9c4ff';
    for (const [a, b] of HAND_CONNECTIONS) {
      octx.beginPath();
      octx.moveTo(points[a].x, points[a].y);
      octx.lineTo(points[b].x, points[b].y);
      octx.stroke();
    }

    // 엄지와 검지 끝을 제외한 관절은 작고 밝은 점으로 표시합니다.
    for (let i = 0; i < points.length; i++) {
      if (i === 4 || i === 8) continue;
      const p = points[i];
      octx.beginPath();
      octx.arc(p.x, p.y, i === 0 ? 4.5 : 3.5, 0, Math.PI * 2);
      octx.fillStyle = i === 0 ? '#ffffff' : '#c8d8ff';
      octx.fill();
      octx.lineWidth = 1.5;
      octx.strokeStyle = '#25334e';
      octx.stroke();
    }
  }

  function drawTip(point, color, label) {
    octx.save();
    octx.shadowColor = color;
    octx.shadowBlur = 15;
    octx.beginPath();
    octx.arc(point.x, point.y, 10, 0, Math.PI * 2);
    octx.fillStyle = color;
    octx.fill();
    octx.shadowBlur = 0;
    octx.lineWidth = 3;
    octx.strokeStyle = '#ffffff';
    octx.stroke();
    octx.font = '700 12px system-ui, sans-serif';
    octx.textAlign = 'center';
    octx.lineWidth = 3;
    octx.strokeStyle = '#10131b';
    octx.strokeText(label, point.x, point.y - 18);
    octx.fillStyle = '#ffffff';
    octx.fillText(label, point.x, point.y - 18);
    octx.restore();
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
      gestureStatus.textContent = '손 전체를 카메라 화면 안에 보여주세요';
      return;
    }

    const points = hand.map(toStagePoint);
    drawHandSkeleton(points);

    const thumb = points[4];
    const index = points[8];
    // 화면상 2D 거리만 사용하면 손을 눕혔을 때 두 손끝이 겹쳐 보이는 오작동이 생깁니다.
    // MediaPipe의 깊이(z)를 포함한 3D 거리로 실제 손끝 간격을 추정합니다.
    const thumb3D = hand[4];
    const index3D = hand[8];
    const wrist3D = hand[0];
    const middleMcp3D = hand[9];
    const gap3D = Math.hypot(
      thumb3D.x - index3D.x,
      thumb3D.y - index3D.y,
      thumb3D.z - index3D.z
    );
    const palmScale3D = Math.max(0.001, Math.hypot(
      wrist3D.x - middleMcp3D.x,
      wrist3D.y - middleMcp3D.y,
      wrist3D.z - middleMcp3D.z
    ));
    const ratio = gap3D / palmScale3D;
    // 손끝을 이전보다 더 가까이 모아야 그리기가 시작되도록 기준을 낮춥니다.
    // 시작 기준 0.22, 해제 기준 0.32로 미세한 오인식을 줄입니다.
    const shouldPinch = pinching ? ratio < 0.32 : ratio < 0.22;

    octx.beginPath();
    octx.moveTo(thumb.x, thumb.y);
    octx.lineTo(index.x, index.y);
    octx.strokeStyle = shouldPinch ? '#83e6b0' : '#ffffff99';
    octx.lineWidth = shouldPinch ? 3.5 : 2;
    octx.stroke();

    drawTip(thumb, '#ff9f43', '엄지');
    drawTip(index, '#5795ff', '검지');

    if (shouldPinch) {
      if (!pinching) {
        currentStroke = { color: colorPicker.value, size: Number(sizePicker.value), points: [] };
        strokes.push(currentStroke);
      }
      // 손끝의 미세한 흔들림을 줄이면서 검지 끝을 따라 그립니다.
      const drawPoint = index;
      if (lastPoint) {
        const smooth = {
          x: lastPoint.x * 0.42 + drawPoint.x * 0.58,
          y: lastPoint.y * 0.42 + drawPoint.y * 0.58
        };
        currentStroke.points.push(smooth);
        lastPoint = smooth;
      } else {
        currentStroke.points.push(drawPoint);
        lastPoint = drawPoint;
      }
      redraw();
      gestureStatus.textContent = '그리는 중 · 엄지와 검지를 모음';
    } else {
      lastPoint = null;
      currentStroke = null;
      gestureStatus.textContent = '엄지와 검지를 모으면 그려집니다';
    }
    pinching = shouldPinch;
    trackingStatus.textContent = '● 손 관절 21개 추적 중';
    trackingStatus.classList.add('active');
    message.textContent = '손 인식 성공 · 엄지와 검지 끝을 모으면 그려집니다';
  }

  startBtn.addEventListener('click', startCamera);
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