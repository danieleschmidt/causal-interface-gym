import DOMPurify from 'isomorphic-dompurify'

/**
 * Security utilities for sanitizing and validating user input
 */

export const sanitizeHtml = (html: string): string => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'target'],
  })
}

export const sanitizeText = (text: string): string => {
  if (typeof text !== 'string') return ''
  
  return text
    .trim()
    .replace(/[<>\"'&]/g, (char) => {
      const map: Record<string, string> = {
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '&': '&amp;',
      }
      return map[char] || char
    })
    .slice(0, 10000) // Prevent excessive length
}

export const validateApiKey = (key: string): boolean => {
  if (!key || typeof key !== 'string') return false
  
  // Basic format validation - adjust based on your API key format
  const apiKeyRegex = /^[a-zA-Z0-9-_]{20,}$/
  return apiKeyRegex.test(key)
}

export const maskApiKey = (key: string): string => {
  if (!key || key.length < 8) return '****'
  return key.slice(0, 4) + '*'.repeat(key.length - 8) + key.slice(-4)
}

export const validateUrl = (url: string): boolean => {
  try {
    const urlObj = new URL(url)
    return ['http:', 'https:'].includes(urlObj.protocol)
  } catch {
    return false
  }
}

export const isValidJson = (str: string): boolean => {
  try {
    JSON.parse(str)
    return true
  } catch {
    return false
  }
}

export const generateNonce = (): string => {
  const array = new Uint8Array(16)
  crypto.getRandomValues(array)
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('')
}

export const hashString = async (str: string): Promise<string> => {
  const encoder = new TextEncoder()
  const data = encoder.encode(str)
  const hash = await crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hash))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

export const validateFileUpload = (file: File): { isValid: boolean; error?: string } => {
  const maxSize = 10 * 1024 * 1024 // 10MB
  const allowedTypes = [
    'application/json',
    'text/plain',
    'text/csv',
    'application/csv',
  ]

  if (file.size > maxSize) {
    return { isValid: false, error: 'File size exceeds 10MB limit' }
  }

  if (!allowedTypes.includes(file.type)) {
    return { isValid: false, error: 'File type not allowed' }
  }

  return { isValid: true }
}

export const rateLimiter = (maxRequests: number, windowMs: number) => {
  const requests = new Map<string, number[]>()

  return (identifier: string): boolean => {
    const now = Date.now()
    const userRequests = requests.get(identifier) || []
    
    // Remove old requests outside the window
    const recentRequests = userRequests.filter(timestamp => now - timestamp < windowMs)
    
    if (recentRequests.length >= maxRequests) {
      return false // Rate limit exceeded
    }
    
    recentRequests.push(now)
    requests.set(identifier, recentRequests)
    return true
  }
}

export const secureStorage = {
  set: (key: string, value: any): void => {
    try {
      const encrypted = btoa(JSON.stringify(value))
      sessionStorage.setItem(key, encrypted)
    } catch (error) {
      console.error('Failed to store data securely:', error)
    }
  },

  get: (key: string): any => {
    try {
      const encrypted = sessionStorage.getItem(key)
      if (!encrypted) return null
      return JSON.parse(atob(encrypted))
    } catch (error) {
      console.error('Failed to retrieve data securely:', error)
      return null
    }
  },

  remove: (key: string): void => {
    sessionStorage.removeItem(key)
  },

  clear: (): void => {
    sessionStorage.clear()
  },
}

export const contentSecurityPolicy = {
  'default-src': "'self'",
  'script-src': "'self' 'unsafe-inline'",
  'style-src': "'self' 'unsafe-inline' fonts.googleapis.com",
  'font-src': "'self' fonts.gstatic.com",
  'img-src': "'self' data: https:",
  'connect-src': "'self' api.openai.com api.anthropic.com",
  'frame-ancestors': "'none'",
  'base-uri': "'self'",
  'form-action': "'self'",
}

export const validateCspNonce = (nonce: string): boolean => {
  const nonceRegex = /^[a-f0-9]{32}$/
  return nonceRegex.test(nonce)
}

export const securityHeaders = {
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
}