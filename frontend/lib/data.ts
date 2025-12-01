import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import { DocumentRecord } from '@/types/schema';

const DATA_DIR = path.join(process.cwd(), '../SANR-Embed/data/processed');
const GOLD_STANDARD_PATH = path.join(DATA_DIR, 'gold_standard.csv');

// Cache the data in memory to avoid reading file on every request
let cachedData: DocumentRecord[] | null = null;

export async function getDocuments(): Promise<DocumentRecord[]> {
  if (cachedData) {
    return cachedData;
  }

  if (!fs.existsSync(GOLD_STANDARD_PATH)) {
    console.warn(`Gold standard file not found at ${GOLD_STANDARD_PATH}`);
    return [];
  }

  const csvFile = fs.readFileSync(GOLD_STANDARD_PATH, 'utf-8');
  
  return new Promise((resolve, reject) => {
    Papa.parse(csvFile, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        cachedData = results.data as DocumentRecord[];
        resolve(cachedData);
      },
      error: (error: any) => {
        reject(error);
      }
    });
  });
}

export async function getDocumentById(id: string): Promise<DocumentRecord | undefined> {
  const documents = await getDocuments();
  return documents.find(doc => doc.id === id);
}

export async function getDocumentsByFilter(filter: Partial<DocumentRecord>): Promise<DocumentRecord[]> {
  const documents = await getDocuments();
  return documents.filter(doc => {
    return Object.entries(filter).every(([key, value]) => {
      // @ts-ignore
      return doc[key] === value;
    });
  });
}






