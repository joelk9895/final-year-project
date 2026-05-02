"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [data, setData] = useState<{ip: string, timestamp: number} | null>(null);

  const fetchIp = async () => {
    try {
      const res = await fetch("/api/report-ip");
      const json = await res.json();
      setData(json);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchIp();
    const interval = setInterval(fetchIp, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#08080a] text-white flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full bg-white/5 border border-white/10 rounded-3xl p-8 shadow-2xl backdrop-blur-xl">
        <h1 className="text-2xl font-serif mb-8 text-center text-white/90 tracking-wide">Ethereal Charge Pi Locator</h1>
        
        <div className="space-y-6">
          <div className="bg-black/50 rounded-2xl p-6 border border-white/5 shadow-inner">
            <p className="text-white/40 text-xs mb-2 uppercase tracking-[0.2em] font-sans font-semibold">Latest Active IP</p>
            <p className="text-4xl font-mono text-[#2de071] tracking-tight">
              {data ? data.ip : "Fetching..."}
            </p>
          </div>
          
          <div className="flex justify-between items-center text-sm font-sans text-white/40 px-2">
            <p className="tracking-wide">LAST PING</p>
            <p className="font-mono">
              {data?.timestamp ? new Date(data.timestamp * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'}) : "--:--:--"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
