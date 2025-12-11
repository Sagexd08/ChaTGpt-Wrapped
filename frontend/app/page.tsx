import { Metadata } from "next";

export const metadata: Metadata = {
  title: "ChatGPT Wrapped - Your Year in Chat",
};

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-cyber-dark via-cyber-darker to-cyber-dark">
      <div className="flex flex-col items-center justify-center min-h-screen px-4">
        <div className="text-center max-w-2xl mx-auto">
          <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-neon bg-clip-text text-transparent animate-pulse">
            ChatGPT Wrapped
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-300 mb-8">
            Discover your year on ChatGPT through the lens of Spotify Wrapped
          </p>
          
          <div className="space-y-4 mb-12">
            <p className="text-gray-400">
              Upload your ChatGPT conversation export and get a personalized, shareable recap with:
            </p>
            <ul className="text-left inline-block text-gray-300 space-y-2">
              <li className="flex items-center gap-3">
                <span className="text-neon-blue">✓</span> Usage analytics & metrics
              </li>
              <li className="flex items-center gap-3">
                <span className="text-neon-green">✓</span> Mood & personality analysis
              </li>
              <li className="flex items-center gap-3">
                <span className="text-neon-pink">✓</span> Top chat moments
              </li>
              <li className="flex items-center gap-3">
                <span className="text-neon-purple">✓</span> Topic breakdown
              </li>
              <li className="flex items-center gap-3">
                <span className="text-neon-blue">✓</span> Achievement badges
              </li>
            </ul>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="px-8 py-4 bg-gradient-neon text-black font-bold rounded-lg hover:shadow-lg hover:shadow-neon-pink transition-all duration-300 glow">
              Start Your Wrapped
            </button>
            <button className="px-8 py-4 border-2 border-neon-blue text-neon-blue font-bold rounded-lg hover:bg-neon-blue hover:text-black transition-all duration-300">
              Learn More
            </button>
          </div>
          
          <p className="text-sm text-gray-500 mt-8">
            🔒 Privacy First: Your data is never stored on our servers
          </p>
        </div>
      </div>
    </main>
  );
}
