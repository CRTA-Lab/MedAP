<template>
    <UAccordion :items="items" />
</template>

<script setup lang="ts">

useHead({
    title: 'Upute'
})

const items = [{
    label: 'Početna stranica',
    icon: 'i-heroicons-home',
    defaultOpen: true,
    content: 'Na početnoj stranici možete vidjeti pregled svih vaših tvrtki, dokumenata i savjeta. Također možete vidjeti ukupan iznos svih vaših faktura.'
}, {
    label: 'Anotacija slike',
    icon: 'i-heroicons-photo',
    content: 'Opisati kako se radi anotacija.'
}, {
    label: 'Postavke',
    icon: 'i-heroicons-cog',
    content: 'Ova aplikacija osmišljena je na način koji Vama dozvoljava da ju prilagodite kako biste bili maksimalno efikasno i kako bi Vam se vođenje knjiga olakšalo. Stoga, postavke su tu kako bi Vam omogućile prilagodbu aplikacije Vašim potrebama.'
}]  
</script>
