export interface DocumentRecord {
  id: string;
  year: number;
  notary: string;
  rollo: string;
  image_num: string;
  text_original: string;
  label_primary: string;
  label_extended: string;
  wikidata_uri: string;
  text_english?: string;
  text_chinese?: string;
  ocr_noisy?: string;
}

export interface ModelMetric {
  model_name: string;
  model_class: 'Tier 1' | 'Tier 2' | 'Tier 3';
  f1_native: number;
  f1_translated: number;
  delta_bias: number;
  ocr_drop: number;
  embedding_drift: number;
}






