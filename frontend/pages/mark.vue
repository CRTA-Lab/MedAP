<template>
    <UButton>
        Sljedeća slika
    </UButton>
    <p v-if="has_images_to_mark">ovdje je slika</p>
    <p v-else>Nema slika za označavati.</p>
</template>

<script lang="ts">

const has_images_to_mark = ref(false)
</script>