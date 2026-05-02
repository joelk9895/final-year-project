import { NextResponse } from 'next/server';

// Temporary in-memory variable. 
// NOTE: Vercel serverless functions are ephemeral. This will reset randomly!
// For a robust production solution, you should link Vercel KV and uncomment the KV lines.
let latestIpData = { ip: 'Waiting for Pi...', timestamp: 0 };

export async function POST(request: Request) {
    try {
        const body = await request.json();
        
        if (body.ip) {
            latestIpData = {
                ip: body.ip,
                timestamp: body.timestamp || Date.now() / 1000
            };
            
            // To use Vercel KV for true persistence:
            // import { kv } from '@vercel/kv';
            // await kv.set('pi_ip_data', latestIpData);
            
            return NextResponse.json({ success: true, data: latestIpData });
        }
        
        return NextResponse.json({ error: 'No IP provided' }, { status: 400 });
    } catch (e) {
        return NextResponse.json({ error: 'Invalid payload' }, { status: 400 });
    }
}

export async function GET() {
    // To use Vercel KV for true persistence:
    // import { kv } from '@vercel/kv';
    // const data = await kv.get('pi_ip_data') || latestIpData;
    // return NextResponse.json(data);
    
    return NextResponse.json(latestIpData);
}
