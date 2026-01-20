export const Theme = {
  colors: {
    background: {
      primary: '#0A0A0A',     
      secondary: '#1C1C1E',   
      elevated: '#2C2C2E',    
      tertiary: '#38383A',    
    },
    surface: {
      overlay: 'rgba(0, 0, 0, 0.4)',      
      overlayStrong: 'rgba(0, 0, 0, 0.7)',
      glass: 'rgba(255, 255, 255, 0.08)', 
      glassStrong: 'rgba(255, 255, 255, 0.12)', 
    },
    text: {
      primary: '#FFFFFF',     
      secondary: '#98989D',   
      tertiary: '#636366',    
      inverse: '#000000',     
    },
    accent: {
      primary: '#0A84FF',     
      success: '#30D158',     
      warning: '#FF9F0A',     
      error: '#FF453A',       
    },
    border: {
      subtle: 'rgba(255, 255, 255, 0.1)',   
      default: 'rgba(255, 255, 255, 0.2)',  
      strong: 'rgba(255, 255, 255, 0.3)',   
    },
  },
  typography: {
    size: {
      xs: 12,
      sm: 14,
      base: 16,
      lg: 18,
      xl: 20,
      '2xl': 24,
      '3xl': 28,
      '4xl': 32,
      '5xl': 40,
      '6xl': 48,
    },
    weight: {
      regular: '400' as const,
      medium: '500' as const,
      semibold: '600' as const,
      bold: '700' as const,
      heavy: '800' as const,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75,
    },
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 12,
    base: 16,
    lg: 24,
    xl: 32,
    '2xl': 48,
    '3xl': 64,
  },
  radius: {
    xs: 4,
    sm: 8,
    md: 12,
    base: 16,
    lg: 20,
    xl: 24,
    '2xl': 28,
    full: 9999,
  },
  shadows: {
    subtle: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 2,
    },
    medium: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 8,
      elevation: 4,
    },
    strong: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 0.2,
      shadowRadius: 16,
      elevation: 8,
    },
    glow: {
      shadowColor: '#0A84FF',
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.5,
      shadowRadius: 12,
      elevation: 6,
    },
  },
  animation: {
    duration: {
      instant: 100,
      fast: 200,
      normal: 300,
      slow: 400,
      slower: 600,
    },
    easing: {
      spring: {
        damping: 15,
        stiffness: 150,
        mass: 1,
      },
      springBouncy: {
        damping: 10,
        stiffness: 200,
        mass: 0.8,
      },
    },
  },
  layout: {
    safeArea: {
      top: 44,
      bottom: 34,
    },
    height: {
      topBar: 60,
      topBarWithSafeArea: 104, // 60 + 44
      bottomBar: 80,
      bottomBarWithSafeArea: 114, // 80 + 34
      button: {
        small: 36,
        medium: 44,
        large: 56,
      },
      input: 48,
    },
    iconSize: {
      xs: 16,
      sm: 20,
      base: 24,
      lg: 28,
      xl: 32,
      '2xl': 40,
    },
    bottomSheet: {
      collapsed: 0.15, 
      half: 0.5,       
      expanded: 0.85,  
    },
  },
  opacity: {
    disabled: 0.4,
    subtle: 0.6,
    medium: 0.8,
    full: 1,
  },
};

export type ThemeColors = typeof Theme.colors;
export type ThemeSpacing = typeof Theme.spacing;
export type ThemeRadius = typeof Theme.radius;
export type ThemeTypography = typeof Theme.typography;
export type ThemeAnimation = typeof Theme.animation;
export type ThemeLayout = typeof Theme.layout;

export const getShadow = (type: keyof typeof Theme.shadows) => Theme.shadows[type];

export const gradients = {
  darkVertical: ['#0A0A0A', '#1C1C1E', '#2C2C2E'],
  primaryGlow: ['#0A84FF', '#0066CC'],
  successGlow: ['#30D158', '#28A745'],
  overlayVertical: ['transparent', 'rgba(0, 0, 0, 0.8)'],
};
