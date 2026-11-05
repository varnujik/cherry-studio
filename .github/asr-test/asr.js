const s=require('sherpa-onnx-node');const j=require('path').join;const fs=require('fs');
const [dir,wavdir]=process.argv.slice(2);
const t0=Date.now();
const rec=new s.OfflineRecognizer({featConfig:{sampleRate:16000,featureDim:80},modelConfig:{funasrNano:{encoderAdaptor:j(dir,'encoder_adaptor.int8.onnx'),llm:j(dir,'llm.int8.onnx'),embedding:j(dir,'embedding.int8.onnx'),tokenizer:j(dir,'tokenizer'),language:process.env.FLANG||''},tokens:'',numThreads:4,provider:'cpu',debug:0}});
const load=Date.now()-t0;let total=0,audio=0;const tag=process.env.TAG;
for(const f of fs.readdirSync(wavdir).filter(f=>f.endsWith('.wav')).sort()){const w=s.readWave(j(wavdir,f));const st=rec.createStream();st.acceptWaveform({sampleRate:w.sampleRate,samples:w.samples});const t=Date.now();rec.decode(st);total+=Date.now()-t;audio+=w.samples.length/w.sampleRate;console.log('RES',tag,f,'|',rec.getResult(st).text.trim());}
console.log('RES',tag,'LOAD_MS',load,'RTF',(total/1000/audio).toFixed(3));
