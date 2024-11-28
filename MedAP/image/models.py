from django.db import models
from django.contrib.auth.models import User

class Image(models.Model):
    '''
    An image that needs to be segmented/annotated.
    '''

    image = models.ImageField()
    mask = models.ImageField(null=True, blank=True)
    is_segmented = models.BooleanField()
    segmented_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return str(self.image.__str__())
