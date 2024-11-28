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
    content: 'Na početnoj stranici možete vidjeti sve i svasta.'
}, {
    label: 'Anotacija slike',
    icon: 'i-heroicons-photo',
    content: 'Opisati kako se radi anotacija.'
}, {
    label: 'Uređivanje maske / anotacije',
    icon: 'i-heroicons-pencil',
    content: 'Kako urediti anotaciju? Uredivanje boxeva, tocaka,...'
}, {
    label: 'Postavke',
    icon: 'i-heroicons-cog',
    content: 'Ova aplikacija osmišljena je na način koji Vama dozvoljava da ju prilagodite kako biste bili maksimalno efikasno i kako bi Vam se vođenje knjiga olakšalo. Stoga, postavke su tu kako bi Vam omogućile prilagodbu aplikacije Vašim potrebama.'
}, {
    label: 'Brisanje korisnika',
    icon: 'i-heroicons-user',
    content: 'Svaka označena slika ima zapis (ime korisnika) koji je označio tu sliku. Brisanjem korisnika ta se oznaka postavlja na praznu vrijednost (NULL). Međutim, slika ostaje označena.'
}]
</script>
