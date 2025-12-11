import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        neon: {
          pink: '#FF006E',
          blue: '#00D9FF',
          green: '#39FF14',
          purple: '#B100FF',
        },
        cyber: {
          dark: '#0a0e27',
          darker: '#050612',
          accent: '#1a1f3a',
        },
      },
      backgroundImage: {
        'gradient-neon': 'linear-gradient(135deg, #FF006E, #00D9FF, #39FF14)',
        'gradient-cyber': 'radial-gradient(circle, #B100FF, #0a0e27)',
      },
      animation: {
        glow: 'glow 2s ease-in-out infinite',
        'slide-up': 'slide-up 0.5s ease-out',
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        glow: {
          '0%, 100%': { boxShadow: '0 0 5px #FF006E, 0 0 10px #FF006E' },
          '50%': { boxShadow: '0 0 20px #FF006E, 0 0 30px #FF006E' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}

export default config
