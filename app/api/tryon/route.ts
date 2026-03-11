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
    // gemini-1.5-flash has separate free quota: 1500 req/day, 1M tokens/min
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' })

    const prompt = `You are a virtual try-on AI.
I am giving you two images:
1. A photo of a PERSON
2. A CLOTHING item (shirt, dress, jacket, etc.)

Your task: Describe in vivid detail exactly how this person would look wearing that clothing item.
- Describe how the garment fits their body shape, skin tone, and pose
- Describe the colors, patterns, and style as they appear on the person
- Describe realistic wrinkles, shadows, and lighting
- Be specific and photorealistic in your description

Then generate a realistic image of the person wearing the clothing.`

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

    // Gemini returned text description (1.5-flash doesn't generate images, returns text)
    const text = result.response.text()
    return NextResponse.json({
      description: text,
      message: 'Gemini has analyzed how the outfit would look on you!'
    }, { status: 200 })

  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    console.error('[tryon]', msg)
    return NextResponse.json({ error: msg }, { status: 500 })
  }
}
