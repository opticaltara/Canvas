"use client";

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Papa, { ParseResult, ParseError } from 'papaparse';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { AlertCircle } from 'lucide-react';
import { columns, CsvRow } from './columns'; // Import columns definition
import { DataTable } from './data-table'; // Import DataTable component
import { ColumnDef, Row } from '@tanstack/react-table'; // Import Row type
import { DataTableColumnHeader } from '@/components/ui/data-table-column-header'; // Import column header

const CsvViewerContent: React.FC = () => {
  const searchParams = useSearchParams();
  const [csvData, setCsvData] = useState<CsvRow[]>([]); // Use CsvRow type
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [csvColumns, setCsvColumns] = useState<ColumnDef<CsvRow>[]>([]); // State for dynamic columns
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const filePath = searchParams.get('filePath');

  // --- START: Replicated CSV Parsing Logic (Consider moving to a shared util) ---
  const parseCsv = (csvString: string): string[][] | null => {
     if (!csvString || typeof csvString !== 'string') return null;
     
     // Basic check - allows files without newlines if they have commas (single header row)
     if (!csvString.includes(',')) return null; 

     const lines = csvString.trim().split(/\r?\n/);
     if (lines.length === 0) return null;
 
     try {
       return lines.map(line => 
         // Basic CSV split, doesn't handle quotes containing commas yet
         line.split(',').map(cell => cell.trim())
       );
     } catch (e) {
       console.error("CSV parsing failed:", e);
       return null;
     }
   };
   // --- END: Replicated CSV Parsing Logic ---

  useEffect(() => {
    // 1. If we have a filePath, fetch from backend as before
    if (filePath) {
      setIsLoading(true);
      setError(null);

      const apiUrl = `/api/read-file?filePath=${encodeURIComponent(filePath)}`;

      fetch(apiUrl)
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.text();
        })
        .then(csvText => {
          // parse via Papa (same logic moved to helper function)
          handleCsvText(csvText);
        })
        .catch(err => {
          console.error("Failed to fetch CSV:", err);
          setError(`Failed to load file: ${err.message}`);
          setIsLoading(false);
        });

      return; // Exit early because we used filePath
    }

    // 2. Fallback: Try to read CSV content directly from sessionStorage
    const storedCsv = typeof window !== 'undefined' ? sessionStorage.getItem('fullCsvData') : null;

    if (!storedCsv) {
      setError('File path is missing and no CSV data found.');
      setIsLoading(false);
      return;
    }

    // We have stored CSV text; parse it client-side
    handleCsvText(storedCsv);
  }, [filePath]);

  // Helper to parse CSV text and populate state via Papa
  const handleCsvText = (csvText: string) => {
    Papa.parse<CsvRow>(csvText, { // Use CsvRow type directly here
      header: true,
      skipEmptyLines: true,
      complete: (results: ParseResult<CsvRow>) => { // Add type for results
        if (results.errors.length > 0) {
          console.error("CSV parsing errors:", results.errors);
          setError(`Error parsing CSV: ${results.errors[0].message}`);
          setIsLoading(false);
          return;
        }
        const headers = results.meta.fields;
        if (!headers) {
          setError("Could not extract headers from CSV.");
          setIsLoading(false);
          return;
        }
        setCsvHeaders(headers);
        setCsvData(results.data); // No need to cast now

        // --- Dynamically generate columns for DataTable ---
        const generatedColumns: ColumnDef<CsvRow>[] = headers.map((header: string): ColumnDef<CsvRow> => ({ // Add return type
          accessorKey: header,
          // Use DataTableColumnHeader for the header
          header: ({ column }) => (
            <DataTableColumnHeader column={column} title={header} />
          ),
          // Optional: Add basic cell rendering
          cell: ({ row }: { row: Row<CsvRow> }) => <div>{row.getValue(header)}</div>, // Add type for row
          // Enable filtering for all columns by default (can be overridden)
          enableColumnFilter: true,
          // Enable sorting for all columns by default (can be overridden)
          enableSorting: true,
        }));
        setCsvColumns(generatedColumns);
        // --- End dynamic column generation ---

        setIsLoading(false);
      },
      error: (err: Error) => { // Use standard Error type
        console.error("CSV parsing error:", err);
        // Check if it's a ParseError to potentially access specific fields
        if ('code' in err && 'message' in err) { 
            setError(`Failed to parse CSV: ${err.message} (Code: ${(err as any).code})`);
        } else {
            setError(`Failed to parse CSV: ${err.message}`);
        }
        setIsLoading(false);
      }
    });
  };

  const renderFullCsvTable = (data: string[][]): React.ReactNode => {
    if (!data || data.length === 0) return <p className="text-sm text-gray-500">Empty CSV data.</p>;

    const headers = data[0];
    const rows = data.slice(1);

    return (
      // Make table container scrollable vertically and horizontally
      <div className="overflow-auto border rounded-md max-h-[calc(100vh-150px)]"> {/* Adjust max-h as needed */}
        <Table className="min-w-full text-sm">
          <TableHeader className="bg-gray-100 sticky top-0 z-10"> {/* Sticky header */}
            <TableRow>
              {headers.map((header, index) => (
                <TableHead key={index} className="px-3 py-2 font-medium text-gray-700 whitespace-nowrap">{header}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, rowIndex) => (
              <TableRow key={rowIndex} className="hover:bg-gray-50">
                {row.map((cell, cellIndex) => (
                  <TableCell key={cellIndex} className="px-3 py-2 whitespace-nowrap">{cell}</TableCell>
                ))}
              </TableRow>
            ))}
            {rows.length === 0 && (
              <TableRow>
                 <TableCell colSpan={headers.length} className="text-center text-gray-500 py-6">No data rows found.</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    );
  };

  if (isLoading) {
    return <div>Loading CSV data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!csvData.length || !csvColumns.length) { // Check csvColumns as well
    return <div>No data found in the CSV file or no columns defined.</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>CSV Data Viewer</CardTitle>
          <CardDescription>Displaying content of: {filePath}</CardDescription>
        </CardHeader>
        <CardContent>
          {/* --- Replace Tremor Table with shadcn/ui DataTable --- */}
          <DataTable columns={csvColumns} data={csvData} />
          {/* --- End replacement --- */}
        </CardContent>
      </Card>
    </div>
  );
};

// Wrap the component in Suspense in the default export
const FullCsvViewerPage: React.FC = () => {
  return (
    <Suspense fallback={<div>Loading file path...</div>}> {/* Add Suspense wrapper */}
      <CsvViewerContent />
    </Suspense>
  );
};

export default FullCsvViewerPage; 