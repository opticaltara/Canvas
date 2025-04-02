import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Combines multiple class names with Tailwind support
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
