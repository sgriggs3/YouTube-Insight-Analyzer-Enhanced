import { createGlobalStyle } from "styled-components"

export const ModernTheme = createGlobalStyle`
  :root {
    --primary-bg: #121212;
    --secondary-bg: #1e1e1e;
    --accent-color: #bb86fc;
    --text-primary: #ffffff;
    --text-secondary: rgba(255,255,255,0.7);
    --error-color: #cf6679;
    --success-color: #03dac6;
  }

  body {
    background: var(--primary-bg);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  // Mobile optimizations
  @media (max-width: 768px) {
    html {
      font-size: 14px;
    }
  }

  // Performance optimizations
  * {
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
    transform: translateZ(0);
  }

  // Smooth scrolling but disable on reduced motion
  @media (prefers-reduced-motion: no-preference) {
    html {
      scroll-behavior: smooth;
    }
  }

  // Add smooth transitions
  * {
    transition: background-color 0.2s ease,
                color 0.2s ease,
                border-color 0.2s ease,
                opacity 0.2s ease;
  }

  // Add new animation keyframes
  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  // Add animation classes
  .fade-in-up {
    animation: fadeInUp 0.4s ease-out;
  }

  // Improve scrollbar styling
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    background: var(--primary-bg);
  }

  ::-webkit-scrollbar-thumb {
    background: var(--secondary-bg);
    border-radius: 4px;
    
    &:hover {
      background: var(--accent-color);
    }
  }
`
