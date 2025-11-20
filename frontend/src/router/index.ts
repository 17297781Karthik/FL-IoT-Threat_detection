import { createRouter, createWebHistory } from 'vue-router';
import Dashboard from '../views/Dashboard.vue';
import Models from '../views/Models.vue';
import Metrics from '../views/Metrics.vue';
import Events from '../views/Events.vue';
import Configuration from '../views/Configuration.vue';
import Logs from '../views/Logs.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: Dashboard,
    },
    {
      path: '/models',
      name: 'models',
      component: Models,
    },
    {
      path: '/metrics',
      name: 'metrics',
      component: Metrics,
    },
    {
      path: '/events',
      name: 'events',
      component: Events,
    },
    {
      path: '/config',
      name: 'configuration',
      component: Configuration,
    },
    {
      path: '/logs',
      name: 'logs',
      component: Logs,
    },
  ],
});

export default router;
