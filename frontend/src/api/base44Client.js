// Mocked base44 client for local development
export const base44 = {
  auth: {
    me: async () => ({
      id: 'local-user',
      name: 'Local User',
      email: 'user@local',
      role: 'admin'
    }),
    logout: () => {
      console.log('Mocked logout');
      window.location.href = '/';
    },
    redirectToLogin: () => {
      console.log('Mocked redirectToLogin');
    }
  },
  api: {
    get: async (url) => {
      console.log(`Mocked GET to ${url}`);
      return {};
    },
    post: async (url, data) => {
      console.log(`Mocked POST to ${url}`, data);
      return {};
    }
  }
};
