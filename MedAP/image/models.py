from django.db import models

class Image(models.Model):
    '''
    An image that needs to be segmented/annotated.
    '''

    image = models.ImageField()
    #mask = models.ImageField(null=True, blank=True)
    is_segmented = models.BooleanField()

    def __str__(self):
        return str(self.image.__str__())
