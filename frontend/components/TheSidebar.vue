<script setup lang="ts">
const links = [
  [{
    label: 'Pregled',
    icon: 'i-heroicons-chart-bar',
    to: '/'
  }, {
    label: 'Slike',
    icon: 'i-heroicons-photo',
    to: '/images'
  }], [{
    label: 'Postavke',
    icon: 'i-heroicons-cog',
    to: '/settings'
  }, {
    label: 'Upute',
    icon: 'i-heroicons-question-mark-circle',
    to: '/how-to'
  }]
]
</script>

<template>
  <UVerticalNavigation :ui="{
    wrapper: 'fixed top-0 left-0 h-screen w-48 border-s border-gray-200 dark:border-gray-800 space-y-2 bg-white dark:bg-gray-900 overflow-y-auto z-10 pt-2',
    base: 'group block border-s -ms-px leading-6 before:hidden',
    padding: 'p-0 ps-4 pt-1',
    rounded: '',
    font: '',
    ring: '',
    active: 'text-primary-500 dark:text-primary-400 border-current font-semibold',
    inactive: 'border-transparent hover:border-gray-400 dark:hover:border-gray-500 text-gray-700 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-300'
  }" :links="links" />
</template>
