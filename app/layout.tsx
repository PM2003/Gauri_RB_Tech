import type { Metadata } from 'next'
export const metadata: Metadata = {
  title: 'RB_Tech Virtual Try-On',
  description: 'AI-powered virtual try-on by RB_Tech, powered by Google Gemini',
}
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{margin:0,padding:0,background:'#0a0a0f',color:'#e2e8f0',fontFamily:'Segoe UI,sans-serif',minHeight:'100vh'}}>
        {children}
      </body>
    </html>
  )
}
