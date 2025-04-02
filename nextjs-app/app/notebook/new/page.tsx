'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import useNotebookStore from '@/app/lib/store';

export default function NewNotebookPage() {
  const router = useRouter();
  const { createNotebook, currentNotebook } = useNotebookStore();

  // Create a new notebook when this page loads
  useEffect(() => {
    createNotebook('New Investigation');
  }, [createNotebook]);

  // Redirect to the notebook page once created
  useEffect(() => {
    if (currentNotebook?.id) {
      router.push(`/notebook/${currentNotebook.id}`);
    }
  }, [currentNotebook?.id, router]);

  // Loading state
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4">Creating new notebook...</h1>
        <div className="w-16 h-16 border-t-4 border-b-4 border-primary-500 rounded-full animate-spin mx-auto"></div>
      </div>
    </div>
  );
}