import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import { chapterSidebar } from '../scripts/chapter-sidebar.mjs';

export const onRequest = defineRouteMiddleware(async (context, next) => {
  const route = context.locals.starlightRoute;
  route.sidebar = chapterSidebar(route.sidebar);
  await next();
});
