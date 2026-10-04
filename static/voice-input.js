export function setupVoiceInput(speech, field, recordButton, fileInput, status) {
  let recorder = null;
  let stream = null;
  async function transcribe(audio) {
    status.textContent = 'Transcribing locally…';
    try {
      const result = await speech.transcribe(audio);
      field.value = result.text;
      status.textContent = result.text || 'No speech recognized.';
    } catch (error) {
      status.textContent = error.message;
    }
  }
  recordButton.onclick = async () => {
    if (recorder?.state === 'recording') {
      recorder.stop();
      recordButton.textContent = 'Record question';
      return;
    }
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw Error('This browser cannot record audio. Upload an audio file or type text.');
      stream = await navigator.mediaDevices.getUserMedia({audio:true});
      const chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        transcribe(new Blob(chunks, {type:recorder.mimeType || 'audio/webm'}));
      };
      recorder.start();
      recordButton.textContent = 'Stop recording';
      status.textContent = 'Recording…';
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop());
      status.textContent = error.message;
    }
  };
  fileInput.onchange = () => { if (fileInput.files[0]) transcribe(fileInput.files[0]); };
  return () => { if (recorder?.state === 'recording') recorder.stop(); stream?.getTracks().forEach(track => track.stop()); };
}
