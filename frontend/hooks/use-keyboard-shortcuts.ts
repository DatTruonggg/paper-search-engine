import { useEffect } from "react"

type ShortcutMap = {
  [key: string]: () => void
}

export function useKeyboardShortcuts(shortcuts: ShortcutMap) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      const modifiers = []

      if (event.ctrlKey || event.metaKey) modifiers.push("mod")
      if (event.altKey) modifiers.push("alt")
      if (event.shiftKey) modifiers.push("shift")

      const shortcutKey = modifiers.length > 0 ? `${modifiers.join("+")}+${key}` : key

      if (shortcuts[shortcutKey]) {
        event.preventDefault()
        shortcuts[shortcutKey]()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [shortcuts])
}

export function useSearchShortcuts(focusSearchInput: () => void, clearSearch?: () => void) {
  useKeyboardShortcuts({
    "mod+k": focusSearchInput,
    "escape": clearSearch || (() => {}),
  })
}