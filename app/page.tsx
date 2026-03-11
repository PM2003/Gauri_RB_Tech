'use client'
import { useState, useRef } from 'react'

export default function Home() {
  const [personImg, setPersonImg]     = useState<string | null>(null)
  const [garmentImg, setGarmentImg]   = useState<string | null>(null)
  const [personFile, setPersonFile]   = useState<File | null>(null)
  const [garmentFile, setGarmentFile] = useState<File | null>(null)
  const [result, setResult]           = useState<string | null>(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [status, setStatus]           = useState('')
  const personRef  = useRef<HTMLInputElement>(null)
  const garmentRef = useRef<HTMLInputElement>(null)

  const toBase64 = (file: File): Promise<string> =>
    new Promise((res, rej) => {
      const r = new FileReader()
      r.onload = () => res((r.result as string).split(',')[1])
      r.onerror = rej
      r.readAsDataURL(file)
    })

  const handlePerson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setPersonFile(f); setPersonImg(URL.createObjectURL(f))
    setResult(null); setError(null)
  }

  const handleGarment = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    setGarmentFile(f); setGarmentImg(URL.createObjectURL(f))
    setResult(null); setError(null)
  }

  const handleTryOn = async () => {
    if (!personFile || !garmentFile) return
    setLoading(true); setError(null); setResult(null)
    try {
      setStatus('Reading images...')
      const [p64, g64] = await Promise.all([toBase64(personFile), toBase64(garmentFile)])
      setStatus('Sending to Gemini AI...')
      const res = await fetch('/api/tryon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          personImage:  { data: p64, mimeType: personFile.type  || 'image/jpeg' },
          garmentImage: { data: g64, mimeType: garmentFile.type || 'image/jpeg' },
        }),
      })
      setStatus('AI generating your look...')
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Unknown error')
      setResult(json.image)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setLoading(false); setStatus('')
    }
  }

  const reset = () => {
    setPersonImg(null); setGarmentImg(null)
    setPersonFile(null); setGarmentFile(null)
    setResult(null); setError(null); setStatus('')
  }

  /* ── Styles ────────────────────────────────────────────────── */
  const card: React.CSSProperties = { background:'#0c0c16', border:'1px solid #1e1e2e', borderRadius:16, padding:24 }
  const uploadBox = (active: boolean): React.CSSProperties => ({
    border: `2px dashed ${active ? '#00c896' : '#1e1e2e'}`,
    borderRadius:12, minHeight:220, display:'flex', alignItems:'center',
    justifyContent:'center', cursor:'pointer', overflow:'hidden',
  })

  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg) } }
        * { box-sizing: border-box }
        button:hover { opacity: 0.88 }
      `}</style>

      <div style={{maxWidth:1100, margin:'0 auto', padding:'0 24px 60px'}}>

        {/* Header */}
        <div style={{padding:'28px 0 32px', borderBottom:'1px solid #1e1e2e', marginBottom:32,
          display:'flex', alignItems:'center', justifyContent:'space-between', flexWrap:'wrap', gap:12}}>
          <div style={{display:'flex', alignItems:'center', gap:16}}>
            <span style={{fontSize:40}}>🧥</span>
            <div>
              <div style={{fontSize:26, fontWeight:800, color:'#f1f5f9'}}>RB_Tech</div>
              <div style={{fontSize:11, color:'#64748b', letterSpacing:3, textTransform:'uppercase'}}>Virtual Try-On</div>
            </div>
          </div>
          <div style={{fontSize:13, color:'#475569', background:'#0c0c16', padding:'8px 18px',
            borderRadius:20, border:'1px solid #1e1e2e'}}>
            ⚡ Powered by Google Gemini AI
          </div>
        </div>

        {/* Upload Grid */}
        <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(280px,1fr))', gap:24, marginBottom:28}}>

          {/* Person */}
          <div style={card}>
            <div style={{fontSize:12, color:'#00c896', fontWeight:700, letterSpacing:2, fontFamily:'monospace', marginBottom:16}}>
              01 — YOUR PHOTO
            </div>
            <div style={uploadBox(!!personImg)} onClick={() => personRef.current?.click()}>
              <input ref={personRef} type="file" accept="image/*" onChange={handlePerson} style={{display:'none'}}/>
              {personImg
                ? <img src={personImg} alt="person" style={{width:'100%', maxHeight:250, objectFit:'contain', borderRadius:8}}/>
                : <div style={{textAlign:'center', padding:24, color:'#475569'}}>
                    <div style={{fontSize:44, marginBottom:10}}>🤳</div>
                    <div style={{fontSize:15, fontWeight:600, color:'#94a3b8'}}>Upload Your Photo</div>
                    <div style={{fontSize:12, marginTop:6}}>Full-body works best</div>
                  </div>}
            </div>
            {personImg && <div style={{marginTop:10, fontSize:13, color:'#00c896', textAlign:'center'}}>✅ Photo ready</div>}
          </div>

          {/* Garment */}
          <div style={card}>
            <div style={{fontSize:12, color:'#00c896', fontWeight:700, letterSpacing:2, fontFamily:'monospace', marginBottom:16}}>
              02 — CLOTHING ITEM
            </div>
            <div style={uploadBox(!!garmentImg)} onClick={() => garmentRef.current?.click()}>
              <input ref={garmentRef} type="file" accept="image/*" onChange={handleGarment} style={{display:'none'}}/>
              {garmentImg
                ? <img src={garmentImg} alt="garment" style={{width:'100%', maxHeight:250, objectFit:'contain', borderRadius:8}}/>
                : <div style={{textAlign:'center', padding:24, color:'#475569'}}>
                    <div style={{fontSize:44, marginBottom:10}}>👕</div>
                    <div style={{fontSize:15, fontWeight:600, color:'#94a3b8'}}>Upload Clothing Item</div>
                    <div style={{fontSize:12, marginTop:6}}>T-shirt, dress, jacket, etc.</div>
                  </div>}
            </div>
            {garmentImg && <div style={{marginTop:10, fontSize:13, color:'#00c896', textAlign:'center'}}>✅ Outfit ready</div>}
          </div>

          {/* Action */}
          <div style={card}>
            <div style={{fontSize:12, color:'#00c896', fontWeight:700, letterSpacing:2, fontFamily:'monospace', marginBottom:16}}>
              03 — GENERATE
            </div>
            <div style={{marginBottom:20, padding:14, background:'#080810', borderRadius:10, fontSize:13,
              color:'#64748b', lineHeight:1.8}}>
              <div>1️⃣ Upload your full-body photo</div>
              <div>2️⃣ Upload a clothing item</div>
              <div>3️⃣ Click Try It On!</div>
              <div style={{marginTop:8, color:'#334155', fontSize:12}}>
                Gemini AI generates a photorealistic result in ~15s
              </div>
            </div>
            <button
              disabled={!personFile || !garmentFile || loading}
              onClick={handleTryOn}
              style={{
                width:'100%', padding:'15px 0',
                background: (!personFile || !garmentFile || loading)
                  ? '#1e1e2e'
                  : 'linear-gradient(135deg,#00c896,#0ea5e9)',
                color: (!personFile || !garmentFile || loading) ? '#475569' : '#0a0a0f',
                border:'none', borderRadius:12, fontSize:17, fontWeight:800,
                cursor: (!personFile || !garmentFile || loading) ? 'not-allowed' : 'pointer',
                marginBottom:12, transition:'all .2s'
              }}>
              {loading ? '⏳ Generating...' : '✨ Try It On!'}
            </button>
            <button onClick={reset} style={{
              width:'100%', padding:'12px 0', background:'transparent',
              color:'#64748b', border:'1px solid #1e1e2e', borderRadius:12,
              fontSize:14, fontWeight:600, cursor:'pointer'
            }}>🔄 Reset</button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{background:'#ff222222', border:'1px solid #ff444444', borderRadius:12,
            padding:'14px 18px', color:'#fca5a5', marginBottom:20}}>
            ⚠️ {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{textAlign:'center', padding:52, background:'#0c0c16',
            border:'1px solid #1e1e2e', borderRadius:16, marginBottom:24}}>
            <div style={{width:52, height:52, border:'4px solid #1e1e2e',
              borderTop:'4px solid #00c896', borderRadius:'50%',
              animation:'spin 1s linear infinite', margin:'0 auto 20px'}}/>
            <div style={{fontSize:18, fontWeight:700, color:'#f1f5f9'}}>Gemini AI is working...</div>
            <div style={{color:'#00c896', fontSize:14, marginTop:8}}>{status}</div>
            <div style={{color:'#475569', fontSize:13, marginTop:6}}>Usually takes 10–20 seconds</div>
          </div>
        )}

        {/* Result */}
        {result && !loading && (
          <div style={{background:'#0c0c16', border:'1px solid #00c89633',
            borderRadius:16, padding:28, textAlign:'center'}}>
            <div style={{fontSize:12, color:'#00c896', fontWeight:700,
              letterSpacing:2, fontFamily:'monospace', marginBottom:20}}>
              ✅ YOUR AI VIRTUAL LOOK
            </div>
            <img src={result} alt="AI try-on result"
              style={{width:'100%', maxHeight:520, objectFit:'contain', borderRadius:12, marginBottom:20}}/>
            <a href={result} download="rb-tech-virtual-look.png"
              style={{display:'inline-block', padding:'13px 36px', background:'#00c896',
                color:'#0a0a0f', borderRadius:10, fontWeight:800, fontSize:15,
                textDecoration:'none', marginRight:12}}>
              ⬇️ Download Look
            </a>
            <button onClick={reset} style={{
              padding:'13px 28px', background:'transparent', color:'#64748b',
              border:'1px solid #1e1e2e', borderRadius:10, fontSize:14,
              fontWeight:600, cursor:'pointer'
            }}>🔄 Try Another</button>
          </div>
        )}

        {/* Footer */}
        <div style={{marginTop:48, textAlign:'center', fontSize:13, color:'#334155'}}>
          Built with ❤️ by RB_Tech · Powered by Google Gemini · Free to use
        </div>
      </div>
    </>
  )
}
