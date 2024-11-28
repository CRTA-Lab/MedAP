<template>
    <p>image {{ id }}</p>
    <p v-if="images_exist">Nema slika za označavati.</p>
</template>

<script setup lang="ts">
const route = useRoute()

const id = route.params.id

const images_exist = ref(false)
</script>