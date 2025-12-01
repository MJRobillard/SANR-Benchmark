import { getDocuments } from "@/lib/data";
import EmbeddingPlot from "@/components/embedding-plot";

export default async function EmbeddingPage() {
  // In a real app, we would fetch the pre-calculated coordinates here 
  // or compute them on the backend.
  const documents = await getDocuments();
  const sampleDocs = documents.slice(0, 500); // Limit for performance in this demo

  return (
    <div className="container py-10 px-4 md:px-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Embedding Space Explorer</h1>
        <p className="text-muted-foreground max-w-3xl">
          Visualizing the semantic relationships between 17th-century legal documents. 
          Points that are closer together represent documents with higher semantic similarity 
          according to the model's internal representation.
        </p>
      </div>

      <EmbeddingPlot documents={sampleDocs} />
      
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-muted-foreground">
        <div className="p-4 bg-neutral-100 rounded-lg">
            <h3 className="font-bold text-neutral-900 mb-2">About this Plot</h3>
            <p>
                This visualization uses dimensionality reduction (e.g., UMAP) to project 
                high-dimensional vector embeddings into 2D space.
            </p>
        </div>
        <div className="p-4 bg-neutral-100 rounded-lg">
            <h3 className="font-bold text-neutral-900 mb-2">Clusters</h3>
            <p>
                Look for distinct groupings. Are "Wills" (Testamentos) separating clearly 
                from "Sales" (Ventas)? This indicates the model understands legal distinctions.
            </p>
        </div>
        <div className="p-4 bg-neutral-100 rounded-lg">
            <h3 className="font-bold text-neutral-900 mb-2">Temporal Drift</h3>
            <p>
                Toggle "Color By: Year" to see if 1653 and 1658 documents occupy different 
                regions, suggesting language evolution or notary changes over time.
            </p>
        </div>
      </div>
    </div>
  );
}






