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
        aubergine: {
          950: '#0b0512',
          900: '#140a21',
          850: '#1a0e2b',
          800: '#231339',
          700: '#331c54',
          600: '#482776',
          500: '#6b3ba8',
        },
        marigold: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
        },
        silver: {
          50: '#ffffff',
          100: '#f8fafc',
          200: '#f1f5f9',
          300: '#e2e8f0',
          400: '#cbd5e1',
          500: '#94a3b8',
          600: '#64748b',
        },
        medical: {
          bg: '#0d0716',         // Velvet Deep Aubergine
          surface: '#150c23',    // Rich Aubergine Surface
          card: '#1c102f',       // Elevated Aubergine Card
          hover: '#291742',      // Deep Aubergine Hover
          border: 'rgba(226, 232, 240, 0.12)', // Ice Silver Hairline
          'border-active': '#f59e0b',          // Marigold Active
          cyan: '#f59e0b',       // Mapped to Marigold Gold
          teal: '#e2e8f0',       // Mapped to Ice Silver
          marigold: '#f59e0b',
          silver: '#e2e8f0',
          aubergine: '#1a0e2b',
          blue: '#3b82f6',
          amber: '#f59e0b',
          rose: '#f43f5e',
          emerald: '#10b981',
          text: '#f8fafc',       // Ice Silver Text
          muted: '#cbd5e1',      // Silver Mist Muted
          subtle: '#94a3b8',     // Platinum Subtle
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
        classic: ['Cinzel', 'Playfair Display', 'Georgia', 'serif'],
      }
    },
  },
  plugins: [],
}

