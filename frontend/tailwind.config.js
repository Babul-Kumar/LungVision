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
          bg: '#f1f5f9',              // Luminous Ice Silver Canvas Background
          surface: '#ffffff',         // Porcelain White Card Surface
          card: '#ffffff',            // Pure Crisp Card Surface
          hover: '#f8fafc',           // Ice Silver Hover Surface
          border: '#e2e8f0',          // Hairline Ice Silver Platinum Border
          'border-active': '#d97706', // Marigold Active Border
          cyan: '#d97706',            // Mapped to Warm Marigold Gold
          teal: '#1a082e',            // Deep Aubergine Accent
          marigold: '#d97706',        // Classic Royal Marigold
          silver: '#f1f5f9',          // Ice Silver Base
          aubergine: '#1a082e',       // Imperial Deep Aubergine
          blue: '#d97706',            // Mapped to Marigold Gold
          amber: '#d97706',
          rose: '#e11d48',
          emerald: '#059669',
          text: '#1a082e',            // High-Contrast Deep Aubergine Typography
          muted: '#5e4b75',           // Refined Muted Aubergine
          subtle: '#5e4b75',          // High-Contrast Plum Aubergine
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

