'use client'

import { useOnlineStatus } from '@/hooks/use-online-status'
import { WifiOff, Wifi } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { useState, useEffect } from 'react'

export function OfflineIndicator() {
  const isOnline = useOnlineStatus()
  const [showReconnected, setShowReconnected] = useState(false)
  const [wasOffline, setWasOffline] = useState(false)

  useEffect(() => {
    if (!isOnline) {
      setWasOffline(true)
    } else if (wasOffline && isOnline) {
      setShowReconnected(true)
      const timer = setTimeout(() => {
        setShowReconnected(false)
        setWasOffline(false)
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [isOnline, wasOffline])

  if (!isOnline) {
    return (
      <Alert className="border-amber-300 bg-gradient-to-r from-amber-50 to-orange-50 shadow-lg">
        <div className="p-1 bg-gradient-to-r from-amber-400 to-orange-400 rounded-full">
          <WifiOff className="h-3 w-3 text-white" />
        </div>
        <AlertDescription className="text-amber-800 font-semibold">
          ⚠️ You're currently offline. Some features may not work properly.
        </AlertDescription>
      </Alert>
    )
  }

  if (showReconnected) {
    return (
      <Alert className="border-emerald-300 bg-gradient-to-r from-emerald-50 to-green-50 shadow-lg">
        <div className="p-1 bg-gradient-to-r from-emerald-400 to-green-400 rounded-full">
          <Wifi className="h-3 w-3 text-white" />
        </div>
        <AlertDescription className="text-emerald-800 font-semibold">
          ✨ Connection restored! You're back online.
        </AlertDescription>
      </Alert>
    )
  }

  return null
}