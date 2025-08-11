import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock performance API for tests
Object.defineProperty(window, 'performance', {
  value: {
    now: () => Date.now(),
    measure: vi.fn(),
    mark: vi.fn(),
    clearMarks: vi.fn(),
    clearMeasures: vi.fn(),
    getEntriesByType: vi.fn(() => []),
    getEntriesByName: vi.fn(() => []),
  },
})

// Mock crypto API for tests
Object.defineProperty(window, 'crypto', {
  value: {
    getRandomValues: (arr: any) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256)
      }
      return arr
    },
    randomUUID: () => 'mock-uuid-' + Math.random().toString(36).substr(2, 9),
    subtle: {
      digest: vi.fn().mockResolvedValue(new ArrayBuffer(32)),
    },
  },
})

// Mock IntersectionObserver
class MockIntersectionObserver {
  observe = vi.fn()
  disconnect = vi.fn()
  unobserve = vi.fn()
  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback
  }
  private callback: IntersectionObserverCallback
}

Object.defineProperty(window, 'IntersectionObserver', {
  value: MockIntersectionObserver,
})

// Mock ResizeObserver
class MockResizeObserver {
  observe = vi.fn()
  disconnect = vi.fn()
  unobserve = vi.fn()
}

Object.defineProperty(window, 'ResizeObserver', {
  value: MockResizeObserver,
})

// Mock requestAnimationFrame
Object.defineProperty(window, 'requestAnimationFrame', {
  value: (callback: FrameRequestCallback) => {
    return setTimeout(callback, 16)
  },
})

// Mock cancelAnimationFrame
Object.defineProperty(window, 'cancelAnimationFrame', {
  value: (id: number) => {
    clearTimeout(id)
  },
})

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
}

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

// Mock sessionStorage
const sessionStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn(),
}

Object.defineProperty(window, 'sessionStorage', {
  value: sessionStorageMock,
})

// Mock URL constructor
global.URL = class URL {
  constructor(public href: string) {}
  toString() {
    return this.href
  }
}

// Mock canvas for D3 testing
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  fillRect: vi.fn(),
  clearRect: vi.fn(),
  getImageData: vi.fn(),
  putImageData: vi.fn(),
  createImageData: vi.fn(),
  setTransform: vi.fn(),
  drawImage: vi.fn(),
  save: vi.fn(),
  fillText: vi.fn(),
  restore: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  closePath: vi.fn(),
  stroke: vi.fn(),
  translate: vi.fn(),
  scale: vi.fn(),
  rotate: vi.fn(),
  arc: vi.fn(),
  fill: vi.fn(),
}))

// Cleanup function to run after each test
afterEach(() => {
  vi.clearAllMocks()
  localStorageMock.clear()
  sessionStorageMock.clear()
})