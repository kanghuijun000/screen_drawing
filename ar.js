(() => {
  const $ = id => document.getElementById(id);
  const arStartBtn = $('arStartBtn');
  const arExitBtn = $('arExitBtn');
  const stage = $('stage');
  const placeholder = $('placeholder');
  const trackingStatus = $('trackingStatus');
  const gestureStatus = $('gestureStatus');
  const cameraLabel = $('cameraLabel');
  const message = $('message');
  const colorPicker = $('colorPicker');
  const sizePicker = $('sizePicker');
  const undoBtn = $('undoBtn');
  const clearBtn = $('clearBtn');

  const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm';
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  const KNOWN_PALM_WIDTH = 0.085;

  let session = null;
  let gl = null;
  let binding = null;
  let landmarker = null;
  let visionApi = null;
  let active = false;
  let strokes = [];
  let currentStroke = null;
  let lastPoint = null;
  let pinching = false;
  let middlePinching = false;
  let middleHoldStartedAt = 0;
  let middleClearTriggered = false;
  let pinchReleaseFrames = 0;
  let lastDetectionAt = 0;
  let cameraWidth = 0;
  let cameraHeight = 0;
  let cameraFramebuffer = null;
  let cameraOutputTexture = null;
  let cameraProgram = null;
  let lineProgram = null;
  let cameraBuffer = null;
  let lineBuffer = null;
  let sourceCanvas = document.createElement('canvas');
  let sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true });
  let cameraPixels = null;
  let flippedPixels = null;
  let cameraMissingReported = false;

  function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const error = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('3D 셰이더 생성 실패: ' + error);
    }
    return shader;
  }

  function createProgram(vertexSource, fragmentSource) {
    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const error = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error('3D 렌더러 연결 실패: ' + error);
    }
    return program;
  }

  function setupGraphics() {
    const canvas = document.createElement('canvas');
    gl = canvas.getContext('webgl2', { alpha: true, antialias: true, depth: true, stencil: false });
    if (!gl) throw new Error('이 브라우저에서 WebGL 2를 사용할 수 없습니다.');
    return gl.makeXRCompatible().then(() => {
      cameraProgram = createProgram(
        '#version 300 es\nin vec2 aPosition;\nout vec2 vUv;\nvoid main(){vUv=(aPosition+1.0)*0.5;gl_Position=vec4(aPosition,0.0,1.0);}',
        '#version 300 es\nprecision highp float;\nuniform sampler2D uCamera;\nin vec2 vUv;\nout vec4 outColor;\nvoid main(){outColor=texture(uCamera,vUv);}'
      );
      lineProgram = createProgram(
        '#version 300 es\nin vec3 aPosition;\nuniform mat4 uViewProjection;\nvoid main(){gl_Position=uViewProjection*vec4(aPosition,1.0);}',
        '#version 300 es\nprecision highp float;\nuniform vec4 uColor;\nout vec4 outColor;\nvoid main(){outColor=uColor;}'
      );
      cameraBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, cameraBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
      lineBuffer = gl.createBuffer();
      cameraFramebuffer = gl.createFramebuffer();
      cameraOutputTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, cameraOutputTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    });
  }

  async function loadHandModel() {
    if (landmarker) return;
    if (!visionApi) {
      visionApi = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs');
    }
    const vision = await visionApi.FilesetResolver.forVisionTasks(WASM_URL);
    landmarker = await visionApi.HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35
    });
  }

  function prepareCameraBuffer(width, height) {
    const targetWidth = Math.min(480, width);
    const targetHeight = Math.max(1, Math.round(targetWidth * height / width));
    if (cameraWidth === targetWidth && cameraHeight === targetHeight && cameraPixels) return;
    cameraWidth = targetWidth;
    cameraHeight = targetHeight;
    sourceCanvas.width = targetWidth;
    sourceCanvas.height = targetHeight;
    cameraPixels = new Uint8Array(targetWidth * targetHeight * 4);
    flippedPixels = new Uint8ClampedArray(targetWidth * targetHeight * 4);
    gl.bindTexture(gl.TEXTURE_2D, cameraOutputTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, targetWidth, targetHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, cameraFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, cameraOutputTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('AR 영상 처리 버퍼를 초기화하지 못했습니다.');
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function captureHandFrame(frame, view, timestamp) {
    const xrCamera = view.camera;
    if (!xrCamera) {
      if (!cameraMissingReported) {
        cameraMissingReported = true;
        message.textContent = '이 기기는 AR 카메라 영상 접근을 지원하지 않습니다. 기존 2D 모드를 사용할 수 있습니다.';
      }
      return;
    }
    const texture = binding.getCameraImage(xrCamera);
    if (!texture) return;
    prepareCameraBuffer(xrCamera.width, xrCamera.height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, cameraFramebuffer);
    gl.viewport(0, 0, cameraWidth, cameraHeight);
    gl.useProgram(cameraProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, cameraBuffer);
    const position = gl.getAttribLocation(cameraProgram, 'aPosition');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(cameraProgram, 'uCamera'), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.readPixels(0, 0, cameraWidth, cameraHeight, gl.RGBA, gl.UNSIGNED_BYTE, cameraPixels);

    const rowBytes = cameraWidth * 4;
    for (let y = 0; y < cameraHeight; y++) {
      const from = (cameraHeight - 1 - y) * rowBytes;
      const to = y * rowBytes;
      flippedPixels.set(cameraPixels.subarray(from, from + rowBytes), to);
    }
    const image = sourceContext.createImageData(cameraWidth, cameraHeight);
    image.data.set(flippedPixels);
    sourceContext.putImageData(image, 0, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const result = landmarker.detectForVideo(sourceCanvas, timestamp);
    processHand(result, view, xrCamera.width, xrCamera.height);
  }

  function cameraPointToWorld(landmark, hand, view, imageWidth, imageHeight) {
    const projection = view.projectionMatrix;
    const palmDx = (hand[5].x - hand[17].x) * imageWidth;
    const palmDy = (hand[5].y - hand[17].y) * imageHeight;
    const palmPixels = Math.max(1, Math.hypot(palmDx, palmDy));
    const focalPixels = Math.abs(projection[0]) * imageWidth * 0.5;
    const baseDepth = Math.max(0.15, Math.min(2.5, focalPixels * KNOWN_PALM_WIDTH / palmPixels));
    const palmNorm = palmPixels / imageWidth;
    const metersPerNorm = KNOWN_PALM_WIDTH / Math.max(0.001, palmNorm);
    const relativeDepth = (landmark.z - hand[0].z) * metersPerNorm;
    const depth = Math.max(0.15, Math.min(2.5, baseDepth + relativeDepth));

    const ndcX = landmark.x * 2 - 1;
    const ndcY = 1 - landmark.y * 2;
    const cameraX = depth * (ndcX + projection[8]) / projection[0];
    const cameraY = depth * (ndcY + projection[9]) / projection[5];
    const cameraZ = -depth;
    const matrix = view.transform.matrix;
    return {
      x: matrix[0] * cameraX + matrix[4] * cameraY + matrix[8] * cameraZ + matrix[12],
      y: matrix[1] * cameraX + matrix[5] * cameraY + matrix[9] * cameraZ + matrix[13],
      z: matrix[2] * cameraX + matrix[6] * cameraY + matrix[10] * cameraZ + matrix[14],
      depth
    };
  }

  function processHand(result, view, imageWidth, imageHeight) {
    const hand = result.landmarks && result.landmarks[0];
    if (!hand || hand.length < 21) {
      pinching = false;
      middlePinching = false;
      middleHoldStartedAt = 0;
      middleClearTriggered = false;
      pinchReleaseFrames = 0;
      lastPoint = null;
      currentStroke = null;
      trackingStatus.textContent = '● 손 인식 대기';
      trackingStatus.classList.remove('active');
      gestureStatus.textContent = '손 전체를 카메라 화면 안에 보여주세요';
      return;
    }

    const thumb = hand[4];
    const index = hand[8];
    const wrist = hand[0];
    const palmBase = hand[9];
    const palmScale = Math.max(0.001, Math.hypot(
      wrist.x - palmBase.x, wrist.y - palmBase.y, wrist.z - palmBase.z
    ));
    const pinchRatio = Math.hypot(
      thumb.x - index.x, thumb.y - index.y, (thumb.z - index.z) * 0.6
    ) / palmScale;
    const middleTip = hand[12];
    const middleRatio = Math.hypot(
      thumb.x - middleTip.x, thumb.y - middleTip.y, (thumb.z - middleTip.z) * 0.45
    ) / palmScale;
    const shouldMiddlePinch = middlePinching ? middleRatio < 0.43 : middleRatio < 0.30;
    const now = performance.now();
    let justReleasedMiddle = false;

    if (shouldMiddlePinch && !middlePinching) {
      middleHoldStartedAt = now;
      middleClearTriggered = false;
      pinching = false;
      lastPoint = null;
      currentStroke = null;
    } else if (!shouldMiddlePinch && middlePinching) {
      middleHoldStartedAt = 0;
      middleClearTriggered = false;
      justReleasedMiddle = true;
    }
    middlePinching = shouldMiddlePinch;

    if (middlePinching && !middleClearTriggered && middleHoldStartedAt && now - middleHoldStartedAt >= 1000) {
      strokes = [];
      currentStroke = null;
      lastPoint = null;
      pinching = false;
      middleClearTriggered = true;
      message.textContent = '3D 공간의 그림을 모두 지웠습니다.';
      gestureStatus.textContent = '전체 지우기 완료 · 엄지와 중지를 1초간 유지';
    }

    const startThreshold = 0.25;
    const releaseThreshold = 0.39;
    let shouldPinch = pinching ? pinchRatio < releaseThreshold : pinchRatio < startThreshold;
    if (pinching && pinchRatio >= releaseThreshold) {
      pinchReleaseFrames++;
      if (pinchReleaseFrames < 3) shouldPinch = true;
    } else {
      pinchReleaseFrames = 0;
    }

    if (middlePinching || justReleasedMiddle) {
      pinching = false;
      lastPoint = null;
      currentStroke = null;
      if (middlePinching && !middleClearTriggered) {
        const remaining = Math.max(0, 1 - (now - middleHoldStartedAt) / 1000);
        gestureStatus.textContent = '전체 지우기까지 ' + remaining.toFixed(1) + '초';
      } else if (!middleClearTriggered) {
        gestureStatus.textContent = '엄지와 검지를 모으면 3D로 그려집니다';
      }
    } else if (shouldPinch) {
      const worldPoint = cameraPointToWorld(index, hand, view, imageWidth, imageHeight);
      if (!pinching) {
        currentStroke = {
          color: colorPicker.value,
          size: Number(sizePicker.value),
          points: []
        };
        strokes.push(currentStroke);
      }
      if (!lastPoint || Math.hypot(
        worldPoint.x - lastPoint.x,
        worldPoint.y - lastPoint.y,
        worldPoint.z - lastPoint.z
      ) > 0.0015) {
        const smoothPoint = lastPoint ? {
          x: lastPoint.x * 0.38 + worldPoint.x * 0.62,
          y: lastPoint.y * 0.38 + worldPoint.y * 0.62,
          z: lastPoint.z * 0.38 + worldPoint.z * 0.62,
          depth: worldPoint.depth
        } : worldPoint;
        currentStroke.points.push(smoothPoint);
        lastPoint = smoothPoint;
      }
      gestureStatus.textContent = '3D 공간에 그리는 중';
    } else {
      lastPoint = null;
      currentStroke = null;
      gestureStatus.textContent = '엄지와 검지를 모으면 3D로 그려집니다';
    }

    pinching = (middlePinching || justReleasedMiddle) ? false : shouldPinch;
    trackingStatus.textContent = '● 3D 손 추적 중';
    trackingStatus.classList.add('active');
  }

  function multiplyMatrices(a, b) {
    const out = new Float32Array(16);
    for (let column = 0; column < 4; column++) {
      for (let row = 0; row < 4; row++) {
        out[column * 4 + row] =
          a[row] * b[column * 4] +
          a[4 + row] * b[column * 4 + 1] +
          a[8 + row] * b[column * 4 + 2] +
          a[12 + row] * b[column * 4 + 3];
      }
    }
    return out;
  }

  function normalize(x, y, z) {
    const length = Math.hypot(x, y, z) || 1;
    return [x / length, y / length, z / length];
  }

  function hexColor(value) {
    const clean = value.replace('#', '');
    return [
      parseInt(clean.slice(0, 2), 16) / 255,
      parseInt(clean.slice(2, 4), 16) / 255,
      parseInt(clean.slice(4, 6), 16) / 255,
      1
    ];
  }

  function makeRibbon(stroke, cameraPosition, focalPixels) {
    const vertices = [];
    for (let i = 1; i < stroke.points.length; i++) {
      const a = stroke.points[i - 1];
      const b = stroke.points[i];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dz = b.z - a.z;
      if (Math.hypot(dx, dy, dz) < 0.0001) continue;
      const direction = normalize(dx, dy, dz);
      const midX = (a.x + b.x) * 0.5;
      const midY = (a.y + b.y) * 0.5;
      const midZ = (a.z + b.z) * 0.5;
      const toCamera = normalize(
        cameraPosition.x - midX,
        cameraPosition.y - midY,
        cameraPosition.z - midZ
      );
      let side = normalize(
        direction[1] * toCamera[2] - direction[2] * toCamera[1],
        direction[2] * toCamera[0] - direction[0] * toCamera[2],
        direction[0] * toCamera[1] - direction[1] * toCamera[0]
      );
      if (Math.hypot(side[0], side[1], side[2]) < 0.1) {
        side = normalize(-direction[1], direction[0], 0);
        if (Math.hypot(side[0], side[1], side[2]) < 0.1) side = [1, 0, 0];
      }
      const distance = Math.hypot(
        cameraPosition.x - midX,
        cameraPosition.y - midY,
        cameraPosition.z - midZ
      );
      const halfWidth = Math.max(0.0004, stroke.size * distance / (2 * Math.max(1, focalPixels)));
      const sx = side[0] * halfWidth;
      const sy = side[1] * halfWidth;
      const sz = side[2] * halfWidth;
      const ap = [a.x + sx, a.y + sy, a.z + sz];
      const am = [a.x - sx, a.y - sy, a.z - sz];
      const bp = [b.x + sx, b.y + sy, b.z + sz];
      const bm = [b.x - sx, b.y - sy, b.z - sz];
      vertices.push(...ap, ...am, ...bp, ...bp, ...am, ...bm);
    }
    return new Float32Array(vertices);
  }

  function renderStrokes(view) {
    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const viewMatrix = view.transform.inverse.matrix;
    const viewProjection = multiplyMatrices(view.projectionMatrix, viewMatrix);
    const cameraPosition = view.transform.position;
    const focalPixels = Math.abs(view.projectionMatrix[0]) * viewport.width * 0.5;

    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uViewProjection'), false, viewProjection);
    const position = gl.getAttribLocation(lineProgram, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);

    for (const stroke of strokes) {
      if (stroke.points.length < 2) continue;
      const vertices = makeRibbon(stroke, cameraPosition, focalPixels);
      if (!vertices.length) continue;
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
      const color = hexColor(stroke.color);
      gl.uniform4f(gl.getUniformLocation(lineProgram, 'uColor'), color[0], color[1], color[2], color[3]);
      gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 3);
    }
  }

  function onXRFrame(timestamp, frame) {
    if (!active || !session) return;
    session.requestAnimationFrame(onXRFrame);
    const pose = frame.getViewerPose(referenceSpace);
    if (!pose || !pose.views.length) return;
    const view = pose.views[0];

    if (landmarker && timestamp - lastDetectionAt >= 55) {
      lastDetectionAt = timestamp;
      try {
        captureHandFrame(frame, view, timestamp);
      } catch (error) {
        console.error('AR 손 인식 오류:', error);
        message.textContent = 'AR 손 인식 처리 오류: ' + (error.message || '알 수 없는 오류');
      }
    }
    renderStrokes(view);
  }

  let referenceSpace = null;

  async function startAR() {
    if (active) return;
    arStartBtn.disabled = true;
    message.textContent = 'AR 공간 추적을 준비하고 있습니다…';
    try {
      if (!navigator.xr || typeof navigator.xr.requestSession !== 'function') {
        throw new Error('이 브라우저는 WebXR AR을 지원하지 않습니다.');
      }
      const nextSession = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['camera-access', 'dom-overlay'],
        domOverlay: { root: document.body }
      });
      session = nextSession;
      await setupGraphics();
      session.updateRenderState({
        baseLayer: new XRWebGLLayer(session, gl, { alpha: true, antialias: true, depth: true })
      });
      referenceSpace = await session.requestReferenceSpace('local');
      if (typeof XRWebGLBinding === 'undefined') {
        throw new Error('이 Chrome에서는 AR 카메라 영상 접근 기능을 사용할 수 없습니다.');
      }
      binding = new XRWebGLBinding(session, gl);
      if (typeof binding.getCameraImage !== 'function') {
        throw new Error('이 Chrome에서는 AR 카메라 영상 접근 기능을 사용할 수 없습니다.');
      }
      active = true;
      cameraMissingReported = false;
      placeholder.classList.add('hidden');
      document.body.classList.add('ar-active');
      arExitBtn.hidden = false;
      cameraLabel.textContent = '3D AR 공간 추적 중';
      trackingStatus.textContent = '● AR 공간 추적 중';
      gestureStatus.textContent = '손 인식 AI 모델을 불러오는 중…';
      message.textContent = 'AR 공간 추적 연결됨 · 손 인식 모델을 불러오는 중…';
      session.addEventListener('end', finishAR, { once: true });
      session.requestAnimationFrame(onXRFrame);
      await loadHandModel();
      gestureStatus.textContent = '엄지와 검지를 모으면 3D로 그려집니다';
      message.textContent = '손을 카메라에 가까이 또는 멀리 움직여 깊이를 바꿔보세요.';
    } catch (error) {
      console.error('3D AR 시작 실패:', error);
      if (session) {
        try { await session.end(); } catch (_) {}
      }
      finishAR();
      message.textContent = error && error.message
        ? '3D AR을 시작하지 못했습니다: ' + error.message + ' 기존 2D 모드는 그대로 사용할 수 있습니다.'
        : '3D AR을 시작하지 못했습니다. 기존 2D 모드를 사용할 수 있습니다.';
    } finally {
      arStartBtn.disabled = false;
    }
  }

  function finishAR() {
    active = false;
    session = null;
    binding = null;
    referenceSpace = null;
    currentStroke = null;
    lastPoint = null;
    pinching = false;
    middlePinching = false;
    middleHoldStartedAt = 0;
    middleClearTriggered = false;
    document.body.classList.remove('ar-active');
    arExitBtn.hidden = true;
    placeholder.classList.remove('hidden');
    cameraLabel.textContent = '후면 카메라 · AI 손 인식 준비 중';
    trackingStatus.textContent = '● 대기 중';
    trackingStatus.classList.remove('active');
    gestureStatus.textContent = '손가락을 카메라에 보여주세요';
    if (undoBtn) undoBtn.disabled = strokes.length === 0;
  }

  arStartBtn.addEventListener('click', startAR);
  arExitBtn.addEventListener('click', () => {
    if (session) session.end();
  });
  undoBtn.addEventListener('click', () => {
    if (!active) return;
    strokes.pop();
    undoBtn.disabled = strokes.length === 0;
  });
  clearBtn.addEventListener('click', () => {
    if (!active) return;
    strokes = [];
    currentStroke = null;
    lastPoint = null;
    undoBtn.disabled = true;
    message.textContent = '3D 공간의 그림을 모두 지웠습니다.';
  });
})();