<template>
  <div>
    <NuxtRouteAnnouncer />
    <NuxtLayout />
  </div>
</template>

<script setup lang="ts">

useHead({
  titleTemplate: (titleChunk) => {
    return titleChunk ? `${titleChunk}` : 'MedAP';
  }
})

</script>

<style>

.main-content {
  margin-left: 12rem; /* Same as sidebar width */
  padding: 2em;
}
</style>