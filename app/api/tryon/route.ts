import { GoogleGenerativeAI } from '@google/generative-ai'
import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GEMINI_API_KEY
    if (!apiKey) return NextResponse.json({ error: 'GEMINI_API_KEY not configured on server' }, { status: 500 })

    const { personImage, garmentImage } = await req.json()
    if (!personImage?.data || !garmentImage?.data)
      return NextResponse.json({ error: 'Both images are required' }, { status: 400 })

    const genAI = new GoogleGenerativeAI(apiKey)
    // Use the stable image-generation capable model
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' })

    const prompt = `You are a virtual try-on AI.
I am giving you two images:
1. A photo of a PERSON
2. A CLOTHING item (shirt, dress, jacket, etc.)

Generate a single realistic photo of that exact person wearing that exact clothing item.
- Keep the person's face, body shape, skin tone and pose exactly the same
- Naturally fit the garment to the person's body with realistic wrinkles/shadows
- Match lighting between the person and the garment
- Output ONLY the final try-on image, no text`

    const result = await model.generateContent([
      { text: prompt },
      { inlineData: { mimeType: personImage.mimeType,  data: personImage.data  } },
      { inlineData: { mimeType: garmentImage.mimeType, data: garmentImage.data } },
    ])

    const parts = result.response.candidates?.[0]?.content?.parts ?? []

    // Look for an image part in the response
    for (const part of parts) {
      if ('inlineData' in part && part.inlineData) {
        const { mimeType, data } = part.inlineData
        return NextResponse.json({ image: `data:${mimeType};base64,${data}` })
      }
    }

    // Gemini returned text — model doesn't support image output on this tier
    const text = result.response.text()
    return NextResponse.json({
      error: 'Gemini described the result instead of generating an image. This usually means the free tier limit was hit. Try again in a minute.',
      details: text.substring(0, 300)
    }, { status: 422 })

  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    console.error('[tryon]', msg)
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
