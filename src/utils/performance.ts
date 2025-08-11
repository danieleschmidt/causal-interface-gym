import { useCallback, useEffect, useMemo, useRef } from 'react'

/**
 * Performance utilities for optimization
 */

// Debounce hook for expensive operations
export const useDebounce = <T extends any[]>(
  callback: (...args: T) => void,
  delay: number
): ((...args: T) => void) => {
  const timeoutRef = useRef<NodeJS.Timeout>()

  return useCallback(
    (...args: T) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      timeoutRef.current = setTimeout(() => callback(...args), delay)
    },
    [callback, delay]
  )
}

// Throttle hook for high-frequency events
export const useThrottle = <T extends any[]>(
  callback: (...args: T) => void,
  delay: number
): ((...args: T) => void) => {
  const lastExecuted = useRef<number>(0)
  const timeoutRef = useRef<NodeJS.Timeout>()

  return useCallback(
    (...args: T) => {
      const now = Date.now()

      if (now - lastExecuted.current >= delay) {
        callback(...args)
        lastExecuted.current = now
      } else {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
        }
        timeoutRef.current = setTimeout(() => {
          callback(...args)
          lastExecuted.current = Date.now()
        }, delay - (now - lastExecuted.current))
      }
    },
    [callback, delay]
  )
}

// Memoized expensive calculations
export const useMemoizedCalculation = <T, P>(
  calculation: (params: P) => T,
  params: P,
  deps: React.DependencyList = []
): T => {
  return useMemo(() => calculation(params), [params, ...deps])
}

// Virtual scrolling for large lists
export const useVirtualList = <T>(
  items: T[],
  itemHeight: number,
  containerHeight: number
) => {
  const scrollTop = useRef(0)
  const visibleRange = useMemo(() => {
    const start = Math.floor(scrollTop.current / itemHeight)
    const visibleCount = Math.ceil(containerHeight / itemHeight)
    const end = Math.min(start + visibleCount + 5, items.length) // Add buffer

    return { start, end, visibleItems: items.slice(start, end) }
  }, [items, itemHeight, containerHeight, scrollTop.current])

  const onScroll = useCallback((event: React.UIEvent<HTMLElement>) => {
    scrollTop.current = event.currentTarget.scrollTop
  }, [])

  return { ...visibleRange, onScroll, totalHeight: items.length * itemHeight }
}

// Performance monitoring
export class PerformanceMonitor {
  private static metrics: Map<string, number[]> = new Map()

  static measure<T>(name: string, fn: () => T): T {
    const start = performance.now()
    const result = fn()
    const duration = performance.now() - start

    if (!this.metrics.has(name)) {
      this.metrics.set(name, [])
    }
    
    const measurements = this.metrics.get(name)!
    measurements.push(duration)
    
    // Keep only last 100 measurements
    if (measurements.length > 100) {
      measurements.shift()
    }

    return result
  }

  static async measureAsync<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now()
    const result = await fn()
    const duration = performance.now() - start

    if (!this.metrics.has(name)) {
      this.metrics.set(name, [])
    }
    
    const measurements = this.metrics.get(name)!
    measurements.push(duration)
    
    if (measurements.length > 100) {
      measurements.shift()
    }

    return result
  }

  static getMetrics(name: string) {
    const measurements = this.metrics.get(name) || []
    if (measurements.length === 0) return null

    const sum = measurements.reduce((a, b) => a + b, 0)
    const avg = sum / measurements.length
    const min = Math.min(...measurements)
    const max = Math.max(...measurements)

    return { avg, min, max, count: measurements.length }
  }

  static getAllMetrics() {
    const result: Record<string, ReturnType<typeof this.getMetrics>> = {}
    for (const [name] of this.metrics) {
      result[name] = this.getMetrics(name)
    }
    return result
  }

  static clearMetrics(name?: string) {
    if (name) {
      this.metrics.delete(name)
    } else {
      this.metrics.clear()
    }
  }
}

// Intersection Observer hook for lazy loading
export const useIntersectionObserver = (
  callback: (isIntersecting: boolean) => void,
  options: IntersectionObserverInit = {}
) => {
  const targetRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const target = targetRef.current
    if (!target) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        callback(entry.isIntersecting)
      },
      {
        threshold: 0.1,
        ...options,
      }
    )

    observer.observe(target)

    return () => {
      observer.unobserve(target)
    }
  }, [callback, options])

  return targetRef
}

// Resource preloader
export class ResourcePreloader {
  private static loadedResources = new Set<string>()
  private static loadingPromises = new Map<string, Promise<void>>()

  static async preloadImage(src: string): Promise<void> {
    if (this.loadedResources.has(src)) return

    if (this.loadingPromises.has(src)) {
      return this.loadingPromises.get(src)!
    }

    const promise = new Promise<void>((resolve, reject) => {
      const img = new Image()
      img.onload = () => {
        this.loadedResources.add(src)
        this.loadingPromises.delete(src)
        resolve()
      }
      img.onerror = () => {
        this.loadingPromises.delete(src)
        reject(new Error(`Failed to load image: ${src}`))
      }
      img.src = src
    })

    this.loadingPromises.set(src, promise)
    return promise
  }

  static async preloadScript(src: string): Promise<void> {
    if (this.loadedResources.has(src)) return

    if (this.loadingPromises.has(src)) {
      return this.loadingPromises.get(src)!
    }

    const promise = new Promise<void>((resolve, reject) => {
      const script = document.createElement('script')
      script.onload = () => {
        this.loadedResources.add(src)
        this.loadingPromises.delete(src)
        resolve()
      }
      script.onerror = () => {
        this.loadingPromises.delete(src)
        reject(new Error(`Failed to load script: ${src}`))
      }
      script.src = src
      document.head.appendChild(script)
    })

    this.loadingPromises.set(src, promise)
    return promise
  }
}

// Memory usage monitor
export const useMemoryMonitor = () => {
  const checkMemoryUsage = useCallback(() => {
    if ('memory' in performance) {
      const memory = (performance as any).memory
      return {
        usedJSHeapSize: memory.usedJSHeapSize,
        totalJSHeapSize: memory.totalJSHeapSize,
        jsHeapSizeLimit: memory.jsHeapSizeLimit,
        usagePercentage: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100,
      }
    }
    return null
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      const usage = checkMemoryUsage()
      if (usage && usage.usagePercentage > 90) {
        console.warn('High memory usage detected:', usage)
      }
    }, 10000) // Check every 10 seconds

    return () => clearInterval(interval)
  }, [checkMemoryUsage])

  return { checkMemoryUsage }
}

// Cache with TTL
export class TTLCache<K, V> {
  private cache = new Map<K, { value: V; expiry: number }>()
  private defaultTTL: number

  constructor(defaultTTL: number = 5 * 60 * 1000) { // 5 minutes default
    this.defaultTTL = defaultTTL
  }

  set(key: K, value: V, ttl: number = this.defaultTTL): void {
    const expiry = Date.now() + ttl
    this.cache.set(key, { value, expiry })
  }

  get(key: K): V | undefined {
    const item = this.cache.get(key)
    if (!item) return undefined

    if (Date.now() > item.expiry) {
      this.cache.delete(key)
      return undefined
    }

    return item.value
  }

  has(key: K): boolean {
    const item = this.cache.get(key)
    if (!item) return false

    if (Date.now() > item.expiry) {
      this.cache.delete(key)
      return false
    }

    return true
  }

  delete(key: K): boolean {
    return this.cache.delete(key)
  }

  clear(): void {
    this.cache.clear()
  }

  cleanup(): number {
    const now = Date.now()
    let cleaned = 0

    for (const [key, item] of this.cache.entries()) {
      if (now > item.expiry) {
        this.cache.delete(key)
        cleaned++
      }
    }

    return cleaned
  }

  size(): number {
    this.cleanup()
    return this.cache.size
  }
}