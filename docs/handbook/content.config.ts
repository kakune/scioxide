import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { createOxiquillCollections } from 'oxiquill/content';

export const collections = {
  ...createOxiquillCollections({ defineCollection, docsLoader, docsSchema }),
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() })
};
