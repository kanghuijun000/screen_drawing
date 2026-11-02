(() => {
  const video=document.getElementById('video'), stage=document.getElementById('stage');
  const drawCanvas=document.getElementById('drawCanvas'), overlayCanvas=document.getElementById('overlayCanvas');
  const ctx=drawCanvas.getContext('2d'), octx=overlayCanvas.getContext('2d');
  const startBtn=document.getElementById('startBtn'), switchBtn=document.getElementById('switchBtn');
  const placeholder=document.getElementById('placeholder'), cameraLabel=document.getElementById('cameraLabel');
  const trackingStatus=document.getElementById('trackingStatus'), gestureStatus=document.getElementById('gestureStatus');
  const message=document.getElementById('message'), colorPicker=document.getElementById('colorPicker'), sizePicker=document.getElementById('sizePicker');
  const undoBtn=document.getElementById('undoBtn'), clearBtn=document.getElementById('clearBtn');
  let stream=null, hands=null, camera='user', running=false, processing=false, pinching=false, lastPoint=null, strokes=[], currentStroke=null, mirrored=true, lastSeen=0;
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  function resize(){
    const rect=stage.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2);
    const old=document.createElement('canvas'); old.width=drawCanvas.width; old.height=drawCanvas.height;
    if(old.width&&old.height) old.getContext('2d').drawImage(drawCanvas,0,0);
    [drawCanvas,overlayCanvas].forEach(c=>{c.width=Math.round(rect.width*dpr);c.height=Math.round(rect.height*dpr);});
    ctx.setTransform(dpr,0,0,dpr,0,0);octx.setTransform(dpr,0,0,dpr,0,0);
    if(old.width)ctx.drawImage(old,0,0,old.width,old.height,0,0,rect.width,rect.height);
  }
  function redraw(){
    const r=stage.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);
    for(const stroke of strokes){if(stroke.points.length<2)continue;ctx.beginPath();ctx.strokeStyle=stroke.color;ctx.lineWidth=stroke.size;ctx.lineCap='round';ctx.lineJoin='round';ctx.moveTo(stroke.points[0].x,stroke.points[0].y);for(let i=1;i<stroke.points.length;i++)ctx.lineTo(stroke.points[i].x,stroke.points[i].y);ctx.stroke();}
    undoBtn.disabled=!strokes.length;
  }
  function stopStream(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}video.srcObject=null;running=false;}
  async function startCamera(){
    message.textContent='카메라 연결 중…';
    try{
      stopStream();
      stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:camera},width:{ideal:1280},height:{ideal:720}}});
      video.srcObject=stream;await video.play();
      const track=stream.getVideoTracks()[0], actual=track.getSettings().facingMode;
      mirrored=actual?actual!=='environment':camera==='user';
      video.style.transform=mirrored?'scaleX(-1)':'none';
      drawCanvas.style.transform=mirrored?'scaleX(-1)':'none';
      overlayCanvas.style.transform=mirrored?'scaleX(-1)':'none';
      await new Promise(resolve=>{if(video.videoWidth)resolve();else video.onloadedmetadata=resolve;});
      resize();placeholder.classList.add('hidden');switchBtn.disabled=false;running=true;
      cameraLabel.textContent=mirrored?'전면 카메라':'후면 카메라';
      message.textContent='영상은 기기 안에서 처리됩니다. 엄지와 검지를 모아보세요.';
      if(hands)await hands.close();
      hands=new Hands({locateFile:file=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
      hands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:.65,minTrackingConfidence:.6});
      hands.onResults(onResults);
      processFrame();
    }catch(err){stopStream();placeholder.classList.remove('hidden');cameraLabel.textContent='카메라 오류';message.textContent=err.name==='NotAllowedError'?'카메라 권한을 허용해 주세요.':err.name==='NotFoundError'?'사용할 수 있는 카메라를 찾지 못했습니다.':'카메라를 시작하지 못했습니다. 브라우저 권한과 연결 상태를 확인해 주세요.';console.error(err);}
  }
  async function processFrame(){
    if(!running)return;
    if(!processing&&hands&&video.readyState>=2){processing=true;try{await hands.send({image:video});}catch(e){console.warn(e);}finally{processing=false;}}
    if(running)requestAnimationFrame(processFrame);
  }
  function onResults(results){
    const r=stage.getBoundingClientRect();octx.clearRect(0,0,r.width,r.height);
    if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){
      if(Date.now()-lastSeen>180){pinching=false;lastPoint=null;currentStroke=null;}
      trackingStatus.textContent='● 손 인식 대기';trackingStatus.classList.remove('active');gestureStatus.textContent='손을 카메라 안에 보여주세요';return;
    }
    lastSeen=Date.now();const lm=results.multiHandLandmarks[0];
    const thumb=lm[4], index=lm[8], wrist=lm[0], middle=lm[9];
    const distance=Math.hypot(thumb.x-index.x,thumb.y-index.y);
    const palm=Math.max(.001,Math.hypot(wrist.x-middle.x,wrist.y-middle.y));
    const ratio=distance/palm;
    const shouldPinch=pinching?ratio<.43:ratio<.30;
    const point={x:index.x*r.width,y:index.y*r.height};
    const drawPoint=point;
    // The entire camera stage is mirrored for the front camera, so canvas coordinates remain camera-relative.
    [thumb,index].forEach((p,i)=>{octx.beginPath();octx.arc(p.x*r.width,p.y*r.height,i===0?5:6,0,Math.PI*2);octx.fillStyle=i===0?'#ffbd69':'#8fb0ff';octx.fill();octx.strokeStyle='#10131b';octx.lineWidth=2;octx.stroke();});
    octx.beginPath();octx.moveTo(thumb.x*r.width,thumb.y*r.height);octx.lineTo(index.x*r.width,index.y*r.height);octx.strokeStyle=shouldPinch?'#83e6b0':'#ffffff70';octx.lineWidth=2;octx.stroke();
    if(shouldPinch){
      if(!pinching){currentStroke={color:colorPicker.value,size:Number(sizePicker.value),points:[]};strokes.push(currentStroke);}
      if(lastPoint){const smooth={x:lastPoint.x*.35+drawPoint.x*.65,y:lastPoint.y*.35+drawPoint.y*.65};currentStroke.points.push(smooth);redraw();lastPoint=smooth;}else{currentStroke.points.push(drawPoint);lastPoint=drawPoint;}
      gestureStatus.textContent='그리는 중 · 손가락 간격 '+Math.round(ratio*100)+'%';
      trackingStatus.textContent='● 손 추적 중';trackingStatus.classList.add('active');
    }else{
      lastPoint=null;currentStroke=null;gestureStatus.textContent='엄지와 검지를 모으면 그려집니다';trackingStatus.textContent='● 손 추적 중';trackingStatus.classList.add('active');
    }
    pinching=shouldPinch;
  }
  startBtn.addEventListener('click',startCamera);
  switchBtn.addEventListener('click',async()=>{camera=camera==='user'?'environment':'user';await startCamera();});
  undoBtn.addEventListener('click',()=>{strokes.pop();redraw();});
  clearBtn.addEventListener('click',()=>{strokes=[];currentStroke=null;lastPoint=null;pinching=false;redraw();});
  window.addEventListener('resize',()=>{resize();redraw();});
  if(!navigator.mediaDevices?.getUserMedia){message.textContent='이 브라우저는 카메라 기능을 지원하지 않습니다. HTTPS 주소에서 최신 Chrome으로 열어주세요.';startBtn.disabled=true;}
  if('serviceWorker'in navigator&&location.protocol.startsWith('http'))window.addEventListener('load',()=>navigator.serviceWorker.register('./sw.js').catch(console.warn));
})();