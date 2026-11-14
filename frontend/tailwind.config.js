/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        medical: {
          bg: '#070b14',
          surface: '#0d1424',
          card: '#111b30',
          hover: '#172440',
          border: '#1e293b',
          'border-active': '#0284c7',
          cyan: '#06b6d4',
          teal: '#14b8a6',
          blue: '#3b82f6',
          amber: '#f59e0b',
          rose: '#f43f5e',
          emerald: '#10b981',
          text: '#f8fafc',
          muted: '#94a3b8',
          subtle: '#64748b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      }
    },
  },
  plugins: [],
}

