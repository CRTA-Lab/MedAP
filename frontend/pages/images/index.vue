<template>
    <UButton to="mark">
        Počni označavati
    </UButton>
    <UTable :columns="columns":rows="people" />
</template>

<script setup lang="ts">

const columns = [{
    key: 'id',
    label: 'ID',
    sortable: true
}, {
    key: 'name',
    label: 'Name',
    sortable: true
}, {
    key: 'segmented',
    label: 'Označeno',
    sortable: true,
}]

const people = [{
    id: 1,
    name: 'Lindsay Walton',
    segmented: 'Da',
}, {
    id: 2,
    name: 'Courtney Henry',
    segmented: 'Ne',
}, {
    id: 3,
    name: 'Tom Cook',
    segmented: 'Ne',
}, {
    id: 4,
    name: 'Whitney Francis',
    segmented: 'Ne',
}, {
    id: 5,
    name: 'Leonard Krasner',
  segmented: 'Ne',
}, {
    id: 6,
    name: 'Floyd Miles',
    segmented: 'Ne',
}]

</script>
