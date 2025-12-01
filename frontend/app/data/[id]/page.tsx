import { getDocumentById, getDocuments } from "@/lib/data";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";
import Link from "next/link";
import { notFound } from "next/navigation";

interface PageProps {
  params: {
    id: string;
  };
}

// This is required for static site generation with dynamic routes
export async function generateStaticParams() {
  const documents = await getDocuments();
  return documents.map((doc) => ({
    id: doc.id,
  }));
}

export default async function DocumentPage({ params }: PageProps) {
  // Await params access for Next.js 15/latest compatibility if needed, 
  // though in standard 14/13 it's direct. 
  // Safe to await in newer versions or just access.
  const { id } = await params; 
  
  const doc = await getDocumentById(id);

  if (!doc) {
    return notFound();
  }

  return (
    <div className="container h-[calc(100vh-3.5rem)] max-h-screen flex flex-col py-6">
      <div className="flex items-center gap-4 mb-6 px-4">
        <Link href="/data">
          <Button variant="ghost" size="icon">
            <ChevronLeft className="w-4 h-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2">
            Document {doc.id}
            <Badge variant="secondary">{doc.year}</Badge>
          </h1>
          <p className="text-sm text-muted-foreground">{doc.notary} • Rollo {doc.rollo}</p>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-0 px-4">
        {/* Left Panel: Image Placeholder (Real implementation would use React-Zoom-Pan-Pinch) */}
        <div className="bg-neutral-100 dark:bg-neutral-900 rounded-lg border flex items-center justify-center text-neutral-400">
          <div className="text-center p-6">
             <p>Image Viewer Placeholder</p>
             <p className="text-xs mt-2">Rollo: {doc.rollo}, Image: {doc.image_num}</p>
          </div>
        </div>

        {/* Right Panel: Text & Metadata */}
        <div className="flex flex-col gap-6 overflow-y-auto pr-2">
          
          {/* Primary Label */}
          <div className="p-4 rounded-lg border bg-card">
             <h3 className="text-sm font-medium text-muted-foreground mb-1">Legal Class</h3>
             <div className="flex flex-wrap gap-2">
                <Badge className="text-base">{doc.label_primary}</Badge>
                {doc.label_extended && (
                  <span className="text-sm text-muted-foreground self-center">
                    {doc.label_extended}
                  </span>
                )}
             </div>
          </div>

          {/* Transcriptions */}
          <div className="space-y-4">
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Original Spanish (17th Century)</h3>
              <div className="p-4 rounded-md bg-neutral-50 dark:bg-neutral-900 border font-serif text-lg leading-relaxed">
                {doc.text_original}
              </div>
            </div>

            {doc.text_english && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium">English Translation</h3>
                <div className="p-4 rounded-md bg-neutral-50 dark:bg-neutral-900 border text-sm leading-relaxed text-muted-foreground">
                  {doc.text_english}
                </div>
              </div>
            )}
             
             {doc.text_chinese && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Chinese Translation</h3>
                <div className="p-4 rounded-md bg-neutral-50 dark:bg-neutral-900 border text-sm leading-relaxed text-muted-foreground">
                  {doc.text_chinese}
                </div>
              </div>
            )}
          </div>
          
          {/* Wikidata Links */}
          {doc.wikidata_uri && (
             <div className="text-xs text-muted-foreground">
                <span className="font-semibold">Wikidata:</span> {doc.wikidata_uri}
             </div>
          )}
        </div>
      </div>
    </div>
  );
}






