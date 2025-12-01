import { getDocuments } from "@/lib/data";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Eye } from "lucide-react";

export default async function DigitalArchive() {
  const documents = await getDocuments();

  // Filter to show only a subset initially or implement pagination (for now showing first 50)
  const displayDocs = documents.slice(0, 50);

  return (
    <div className="container py-10 px-4 md:px-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Digital Archive</h1>
        <p className="text-muted-foreground">
          Explore {documents.length} transcribed records from 17th-century Buenos Aires.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {displayDocs.map((doc) => (
          <Card key={doc.id} className="flex flex-col">
            <CardHeader>
              <div className="flex justify-between items-start">
                 <Badge variant="secondary" className="mb-2">{doc.year}</Badge>
                 <Badge variant="outline">{doc.id}</Badge>
              </div>
              <CardTitle className="text-lg line-clamp-1">{doc.label_primary || "Unclassified"}</CardTitle>
              <CardDescription className="line-clamp-1">{doc.notary}</CardDescription>
            </CardHeader>
            <CardContent className="flex-1">
              <div className="bg-neutral-100 dark:bg-neutral-900 p-3 rounded-md text-xs font-mono text-neutral-600 dark:text-neutral-400 h-32 overflow-hidden relative">
                {doc.text_original}
                <div className="absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-neutral-100 to-transparent dark:from-neutral-900" />
              </div>
            </CardContent>
            <CardFooter>
              <Link href={`/data/${doc.id}`} className="w-full">
                <Button className="w-full gap-2" variant="outline">
                  <Eye className="w-4 h-4" /> View Document
                </Button>
              </Link>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}






